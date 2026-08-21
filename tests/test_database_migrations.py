import sqlite3

import pytest

from models import database


@pytest.mark.asyncio
async def test_init_db_migrates_existing_documents_with_indexing_status(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                text_path TEXT NOT NULL,
                token_count INTEGER,
                is_active INTEGER DEFAULT 1,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()

    db = await database.get_db()
    try:
        columns = {
            row["name"]
            for row in await (await db.execute("PRAGMA table_info(documents)")).fetchall()
        }
    finally:
        await db.close()

    assert {"index_status", "index_error", "indexed_at", "chunk_count"} <= columns

    db = await database.get_db()
    try:
        chunk_columns = {
            row["name"]
            for row in await (await db.execute("PRAGMA table_info(document_chunks)")).fetchall()
        }
    finally:
        await db.close()
    assert {"page_start", "page_end"} <= chunk_columns


@pytest.mark.asyncio
async def test_new_document_tracks_indexing_then_ready_state(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()

    document_id = await database.add_document(
        "חוזר בדיקה", "pdf", "source.pdf", "document.txt", 120
    )
    created = await database.get_document(document_id)
    assert created["index_status"] == "indexing"
    assert created["chunk_count"] == 0

    await database.update_document_index_status(
        document_id, "ready", chunk_count=3
    )
    indexed = await database.get_document(document_id)
    assert indexed["index_status"] == "ready"
    assert indexed["index_error"] is None
    assert indexed["chunk_count"] == 3
    assert indexed["indexed_at"] is not None
