import sqlite3

import pytest

from models import database


@pytest.mark.asyncio
async def test_init_db_adds_validity_columns_and_backfills_from_title(tmp_path, monkeypatch):
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
        connection.execute(
            "INSERT INTO documents (id, title, source_type, text_path) VALUES (1, ?, 'pdf', 'doc1.txt')",
            ("חוזר גמל 2024-9-8 שיעורי תמותה",),
        )
        connection.execute(
            "INSERT INTO documents (id, title, source_type, text_path) VALUES (2, ?, 'pdf', 'doc2.txt')",
            ("חוזר ללא תאריך ידוע",),
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
        rows = {
            row["id"]: dict(row)
            for row in await (await db.execute("SELECT id, effective_date FROM documents")).fetchall()
        }
    finally:
        await db.close()

    assert {"effective_date", "valid_until", "superseded_by"} <= columns
    assert rows[1]["effective_date"] == "2024-09-08"
    assert rows[2]["effective_date"] is None
