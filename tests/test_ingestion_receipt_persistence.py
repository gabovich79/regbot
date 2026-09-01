import json

import pytest

from models import database
from services.document_ingestion_service import (
    build_ingestion_receipt,
    persist_ingestion_receipt,
)


TEXT = """
חוזר גופים מוסדיים 2016-9-11
העברת כספים בין קופות גמל - תיקון
טיפול בבקשת העברה
הגוף המנהל יעביר כספים בהתאם לבקשת העמית ולמועדים הקבועים בהוראות.
"""


@pytest.mark.asyncio
async def test_persisted_receipt_reconciles_profile_nodes_and_fts(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regbot.db"))
    await database.init_db()
    document_id = await database.add_document(
        "העברת כספים בין קופות גמל - תיקון",
        "docx",
        "22.docx",
        "22.txt",
        100,
    )
    document = {
        "id": document_id,
        "title": "העברת כספים בין קופות גמל - תיקון",
        "source_type": "docx",
        "source_ref": "22.docx",
        "document_type": "חוזר",
        "lifecycle_status": "current",
    }
    receipt = build_ingestion_receipt(
        document,
        TEXT,
        original_path="/sources/22.docx",
        source_checksum="c" * 64,
    )

    db = await database.get_db()
    try:
        persisted = await persist_ingestion_receipt(
        db,
        receipt,
        node_embeddings_by_hash={
            node["text_hash"]: [0.1, 0.2] for node in receipt["nodes"]
        },
    )
        profile = await (
            await db.execute(
                "SELECT canonical_title, integrity_status FROM document_profiles WHERE document_id = ?",
                (document_id,),
            )
        ).fetchone()
        nodes = await (
            await db.execute(
                "SELECT COUNT(*) AS count FROM document_nodes WHERE document_id = ?",
                (document_id,),
            )
        ).fetchone()
        node_fts = await (
            await db.execute(
                "SELECT COUNT(*) AS count FROM document_nodes_fts WHERE document_id = ?",
                (document_id,),
            )
        ).fetchone()
        embedded_nodes = await (
            await db.execute(
                "SELECT COUNT(*) AS count FROM document_nodes WHERE document_id = ? AND embedding IS NOT NULL",
                (document_id,),
            )
        ).fetchone()
        stored = await (
            await db.execute(
                "SELECT status, receipt_json FROM document_ingestion_receipts WHERE document_id = ?",
                (document_id,),
            )
        ).fetchone()
    finally:
        await db.close()

    assert persisted["profile_records"] == 1
    assert persisted["node_records"] >= 1
    assert profile["canonical_title"] == "העברת כספים בין קופות גמל - תיקון"
    assert profile["integrity_status"] == "verified"
    assert nodes["count"] == persisted["node_records"]
    assert node_fts["count"] == persisted["node_records"]
    assert embedded_nodes["count"] == persisted["embedded_node_records"]
    assert persisted["embedded_node_records"] == persisted["node_records"]
    assert stored["status"] == "validated"
    assert json.loads(stored["receipt_json"])["document_id"] == document_id
