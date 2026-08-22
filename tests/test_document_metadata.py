import sqlite3

import pytest

from models import database


@pytest.mark.asyncio
async def test_update_document_metadata_updates_document_and_chunk_titles(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()

    doc_id = await database.add_document("2021-9-5.pdf", "pdf", "2021-9-5.pdf", "doc.txt", 100)
    db = await database.get_db()
    try:
        await db.execute(
            "INSERT INTO document_chunks (document_id, content, chunk_index, document_title, embedding) VALUES (?, ?, ?, ?, ?)",
            (doc_id, "תוכן", 0, "2021-9-5.pdf", "[]"),
        )
        await db.commit()
    finally:
        await db.close()

    await database.update_document_metadata(
        doc_id,
        title="חוזר גופים מוסדיים 2021-9-5 — הצטרפות לקופת גמל",
        topic="הצטרפות לקרן פנסיה או לקופת גמל",
        document_type="חוזר",
        lifecycle_status="current",
    )

    doc = await database.get_document(doc_id)
    assert doc["title"].startswith("חוזר גופים מוסדיים 2021-9-5")
    assert doc["topic"] == "הצטרפות לקרן פנסיה או לקופת גמל"
    assert doc["document_type"] == "חוזר"
    assert doc["lifecycle_status"] == "current"

    db = await database.get_db()
    try:
        row = await (await db.execute(
            "SELECT document_title FROM document_chunks WHERE document_id = ?", (doc_id,)
        )).fetchone()
    finally:
        await db.close()
    assert row["document_title"] == doc["title"]


@pytest.mark.asyncio
async def test_update_document_metadata_rejects_unknown_lifecycle_status(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()
    doc_id = await database.add_document("מסמך", "pdf", "ref", "doc.txt", 10)

    with pytest.raises(ValueError, match="lifecycle status"):
        await database.update_document_metadata(doc_id, lifecycle_status="maybe")
