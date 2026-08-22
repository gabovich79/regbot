import pytest

from models import database


@pytest.mark.asyncio
async def test_archive_document_keeps_source_but_removes_it_from_active_documents(tmp_path, monkeypatch):
    db_path = tmp_path / "regbot.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    await database.init_db()

    doc_id = await database.add_document("חוק כפול.docx", "doc", "dup.docx", "doc.txt", 100)
    await database.archive_document(doc_id)

    archived = await database.get_document(doc_id)
    assert archived["is_active"] == 0
    assert archived["lifecycle_status"] == "historical"
    assert archived["title"] == "חוק כפול.docx"
    assert await database.get_all_documents(active_only=True) == []
