import sqlite3

import pytest

from models import database
from services.corpus_audit_service import audit_corpus


@pytest.mark.asyncio
async def test_set_document_validity_marks_superseded(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()

    old_id = await database.add_document("חוזר ישן", "pdf", "old.pdf", "old.txt", 100)
    new_id = await database.add_document("חוזר חדש", "pdf", "new.pdf", "new.txt", 100)

    await database.set_document_validity(
        old_id, effective_date="2016-09-08", superseded_by=new_id
    )

    updated = await database.get_document(old_id)
    assert updated["effective_date"] == "2016-09-08"
    assert updated["superseded_by"] == new_id


def test_audit_reports_validity_status(tmp_path):
    db_path = tmp_path / "corpus.db"
    (tmp_path / "old.txt").write_text("תוכן", encoding="utf-8")
    (tmp_path / "new.txt").write_text("תוכן", encoding="utf-8")

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                text_path TEXT NOT NULL,
                original_path TEXT,
                source_checksum TEXT,
                token_count INTEGER,
                is_active INTEGER DEFAULT 1,
                index_status TEXT NOT NULL DEFAULT 'pending',
                index_error TEXT,
                indexed_at DATETIME,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                effective_date TEXT,
                valid_until TEXT,
                superseded_by INTEGER,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
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
                page_start INTEGER,
                page_end INTEGER,
                embedding TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO documents (id, title, source_type, text_path, is_active, chunk_count, effective_date, superseded_by) VALUES (1, 'חוזר ישן', 'pdf', ?, 1, 1, '2016-09-08', 2)",
            (str(tmp_path / "old.txt"),),
        )
        connection.execute(
            "INSERT INTO documents (id, title, source_type, text_path, is_active, chunk_count, effective_date) VALUES (2, 'חוזר חדש', 'pdf', ?, 1, 1, '2024-09-08')",
            (str(tmp_path / "new.txt"),),
        )
        connection.commit()
    finally:
        connection.close()

    report = audit_corpus(db_path)

    by_id = {doc["id"]: doc for doc in report["documents"]}
    assert by_id[1]["validity_status"] == "superseded"
    assert by_id[2]["validity_status"] == "current"
