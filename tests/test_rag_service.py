import json

import aiosqlite
import pytest

from services import rag_service


async def _chunk_db():
    db = await aiosqlite.connect(":memory:")
    db.row_factory = aiosqlite.Row
    await db.execute(
        """
        CREATE TABLE document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            section_header TEXT,
            chunk_index INTEGER NOT NULL,
            document_title TEXT,
            document_ref TEXT,
            effective_date TEXT,
            topic TEXT,
            embedding TEXT NOT NULL
        )
        """
    )
    await db.commit()
    return db


class _FailingEmbeddings:
    async def create(self, **_kwargs):
        raise RuntimeError("embedding provider unavailable")


class _FailingClient:
    embeddings = _FailingEmbeddings()


class _Embedding:
    def __init__(self, values):
        self.embedding = values


class _ShortEmbeddingResponse:
    data = [_Embedding([0.3, 0.4])]


class _ShortEmbeddings:
    async def create(self, **_kwargs):
        return _ShortEmbeddingResponse()


class _ShortResponseClient:
    embeddings = _ShortEmbeddings()


@pytest.mark.asyncio
async def test_embedding_failure_keeps_existing_document_chunks(monkeypatch):
    db = await _chunk_db()
    try:
        await db.execute(
            """
            INSERT INTO document_chunks
            (document_id, content, section_header, chunk_index, document_title,
             document_ref, effective_date, topic, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (7, "הקטע הקיים", "סעיף 1", 0, "מסמך קיים", "ref", None, None, json.dumps([0.1, 0.2])),
        )
        await db.commit()
        monkeypatch.setattr(rag_service, "openai_client", _FailingClient())

        chunks = [
            {
                "document_id": 7,
                "content": "קטע חדש",
                "section_header": "סעיף 2",
                "chunk_index": 0,
                "document_title": "מסמך קיים",
                "document_ref": "ref",
                "effective_date": None,
                "topic": None,
            }
        ]

        with pytest.raises(RuntimeError, match="embedding provider unavailable"):
            await rag_service.embed_and_store_chunks(chunks, db)

        rows = await (await db.execute(
            "SELECT content FROM document_chunks WHERE document_id = ?", (7,)
        )).fetchall()
        assert [row["content"] for row in rows] == ["הקטע הקיים"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_short_embedding_response_keeps_existing_document_chunks(monkeypatch):
    db = await _chunk_db()
    try:
        await db.execute(
            """
            INSERT INTO document_chunks
            (document_id, content, section_header, chunk_index, document_title,
             document_ref, effective_date, topic, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (7, "הקטע הקיים", "סעיף 1", 0, "מסמך קיים", "ref", None, None, json.dumps([0.1, 0.2])),
        )
        await db.commit()
        monkeypatch.setattr(rag_service, "openai_client", _ShortResponseClient())
        chunks = [
            {
                "document_id": 7,
                "content": "קטע חדש ראשון",
                "section_header": "סעיף 2",
                "chunk_index": 0,
                "document_title": "מסמך קיים",
                "document_ref": "ref",
                "effective_date": None,
                "topic": None,
            },
            {
                "document_id": 7,
                "content": "קטע חדש שני",
                "section_header": "סעיף 3",
                "chunk_index": 1,
                "document_title": "מסמך קיים",
                "document_ref": "ref",
                "effective_date": None,
                "topic": None,
            },
        ]

        with pytest.raises(ValueError, match="expected 2 embeddings, got 1"):
            await rag_service.embed_and_store_chunks(chunks, db)

        rows = await (await db.execute(
            "SELECT content FROM document_chunks WHERE document_id = ?", (7,)
        )).fetchall()
        assert [row["content"] for row in rows] == ["הקטע הקיים"]
    finally:
        await db.close()
