import sqlite3

import pytest

from models import database


async def _table_names(db):
    rows = await (
        await db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    ).fetchall()
    return {row["name"] for row in rows}


async def _columns(db, table):
    rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
    return {row["name"] for row in rows}


@pytest.mark.asyncio
async def test_init_db_creates_hierarchical_profile_and_node_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regbot.db"))

    await database.init_db()
    await database.init_db()  # migrations must be idempotent

    db = await database.get_db()
    try:
        tables = await _table_names(db)
        profile_columns = await _columns(db, "document_profiles")
        node_columns = await _columns(db, "document_nodes")
    finally:
        await db.close()

    assert {
        "document_profiles",
        "document_nodes",
        "document_profiles_fts",
        "document_nodes_fts",
    } <= tables

    assert {
        "document_id",
        "canonical_title",
        "official_number",
        "issuer",
        "profile_summary",
        "scope_in_json",
        "scope_out_json",
        "topics_json",
        "keywords_json",
        "heading_outline_json",
        "identity_evidence_json",
        "profile_embedding",
        "integrity_status",
        "integrity_reasons_json",
        "review_status",
        "profile_version",
    } <= profile_columns

    assert {
        "id",
        "document_id",
        "parent_id",
        "node_type",
        "node_path",
        "section_label",
        "heading",
        "raw_text",
        "retrieval_text",
        "page_start",
        "page_end",
        "ordinal",
        "text_hash",
        "embedding",
        "is_evidence",
        "index_version",
    } <= node_columns


@pytest.mark.asyncio
async def test_hierarchical_nodes_enforce_parent_document_consistency(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regbot.db"))
    await database.init_db()

    document_id = await database.add_document(
        "חוק בדיקה", "pdf", "source.pdf", "source.txt", 100
    )
    db = await database.get_db()
    try:
        cursor = await db.execute(
            """
            INSERT INTO document_nodes
                (document_id, parent_id, node_type, node_path, raw_text,
                 retrieval_text, ordinal, text_hash, is_evidence, index_version)
            VALUES (?, NULL, 'section', '25', 'טקסט מקור',
                    'חוק בדיקה | סעיף 25 | טקסט מקור', 0, 'hash-1', 1, 1)
            """,
            (document_id,),
        )
        parent_id = cursor.lastrowid
        await db.execute(
            """
            INSERT INTO document_nodes
                (document_id, parent_id, node_type, node_path, raw_text,
                 retrieval_text, ordinal, text_hash, is_evidence, index_version)
            VALUES (?, ?, 'subsection', '25/א', 'טקסט בן',
                    'חוק בדיקה | סעיף 25(א) | טקסט בן', 1, 'hash-2', 1, 1)
            """,
            (document_id, parent_id),
        )
        await db.commit()

        rows = await (
            await db.execute(
                "SELECT id, document_id, parent_id, node_type FROM document_nodes ORDER BY id"
            )
        ).fetchall()
    finally:
        await db.close()

    assert len(rows) == 2
    assert rows[1]["parent_id"] == rows[0]["id"]
    assert rows[1]["document_id"] == rows[0]["document_id"]

    other_document_id = await database.add_document(
        "חוק אחר", "pdf", "other.pdf", "other.txt", 50
    )
    db = await database.get_db()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="parent document mismatch"):
            await db.execute(
                """
                INSERT INTO document_nodes
                    (document_id, parent_id, node_type, node_path, raw_text,
                     retrieval_text, ordinal, text_hash, is_evidence, index_version)
                VALUES (?, ?, 'subsection', '25/ב', 'שייך למסמך אחר',
                        'חוק אחר | סעיף 25(ב)', 0, 'hash-3', 1, 1)
                """,
                (other_document_id, parent_id),
            )
    finally:
        await db.close()
