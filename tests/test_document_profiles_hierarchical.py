import json

import pytest

from models import database
from services.document_integrity_service import assess_document_integrity
from services.document_profile_service import (
    build_document_profile,
    save_document_profile,
)


TRANSFER_TEXT = """
חוזר גופים מוסדיים 2016-9-11
העברת כספים בין קופות גמל - תיקון
מטרת חוזר זה היא להסדיר את הליכי העברת הכספים בין קופות גמל,
לרבות לוחות זמנים, ביטול בקשה והעברת מידע בין החברות המנהלות.
"""


def document(title, **overrides):
    return {
        "id": 22,
        "title": title,
        "source_type": "url",
        "source_ref": "https://www.gov.il/example.docx",
        "topic": None,
        "document_type": "חוזר",
        "effective_date": "2016-09-11",
        "valid_until": None,
        "lifecycle_status": "current",
        "original_path": None,
        "source_checksum": None,
        **overrides,
    }


def test_integrity_flags_title_body_topic_mismatch():
    doc = document("חוזר גמל 2016-9-11 מסלולי השקעה תלויי גיל")
    profile = build_document_profile(doc, TRANSFER_TEXT)

    result = assess_document_integrity(doc, TRANSFER_TEXT, profile)

    assert result["status"] == "warning"
    assert "title_body_mismatch" in result["reasons"]
    assert "2016-9-11" in profile["official_number"]
    assert any("העברת כספים" in item for item in profile["identity_evidence"])


def test_integrity_accepts_corrected_title_for_same_source():
    doc = document("העברת כספים בין קופות גמל - תיקון")
    profile = build_document_profile(doc, TRANSFER_TEXT)

    result = assess_document_integrity(doc, TRANSFER_TEXT, profile)

    assert "title_body_mismatch" not in result["reasons"]
    assert result["status"] in {"verified", "warning"}
    assert "העברת כספים" in profile["canonical_title"]


def test_integrity_flags_official_number_conflict():
    doc = document("חוזר גופים מוסדיים 2024-9-8", effective_date="2024-09-08")
    profile = build_document_profile(doc, TRANSFER_TEXT)

    result = assess_document_integrity(doc, TRANSFER_TEXT, profile)

    assert result["status"] == "failed"
    assert "official_number_conflict" in result["reasons"]


def test_profile_prefers_number_in_identity_heading_over_referenced_old_circular():
    text = """
חוזר גופים מוסדיים 2022-9-3
ניוד בין קופות גמל - תיקון
בהתאם להוראות חוזר גופים מוסדיים 2009-9-9 ולתקנות ההעברה.
"""
    doc = document("חוזר גמל 2022-9-3 ניוד בין קופות", id=28)

    profile = build_document_profile(doc, text)
    result = assess_document_integrity(doc, text, profile)

    assert profile["official_number"] == "2022-9-3"
    assert "official_number_conflict" not in result["reasons"]


def test_profile_normalizes_filename_separators_for_integrity_overlap():
    text = """
חוזר גופים מוסדיים 2016-9-17
כללי השקעה החלים על גופים מוסדיים
"""
    doc = document(
        "כללי_השקעה_החלים_גופים מוסדיים_2016-9-17.pdf", id=36
    )

    profile = build_document_profile(doc, text)
    result = assess_document_integrity(doc, text, profile)

    assert "title_body_mismatch" not in result["reasons"]


def test_integrity_flags_reversed_hebrew_extraction():
    text = "לכ :דוב \nמ לדג \nמ\"עב חוטיבל הרבח \nממ\"עב למג תופוקו היסנפ תונרק תפקמ לדג"
    doc = document("סיווג הפקדות משרתי מילואים 2024-קבוצתי.pdf", id=9)

    result = assess_document_integrity(doc, text, build_document_profile(doc, text))

    assert "extraction_hebrew_reversed" in result["reasons"]


def test_integrity_flags_binary_extraction():
    binary_text = "\x00" * 200 + "text"
    doc = document("חוזר אימות זהות לקוח", id=27)

    result = assess_document_integrity(doc, binary_text, build_document_profile(doc, binary_text))

    assert "extraction_binary" in result["reasons"]


def test_profile_keeps_retrieval_summary_separate_from_identity_evidence():
    doc = document("העברת כספים בין קופות גמל - תיקון")

    profile = build_document_profile(doc, TRANSFER_TEXT)

    assert profile["profile_summary"]
    assert profile["identity_evidence"]
    assert profile["profile_summary"] not in profile["identity_evidence"]
    assert profile["review_status"] == "machine"


@pytest.mark.asyncio
async def test_save_profile_upserts_profile_and_fts_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regbot.db"))
    await database.init_db()
    document_id = await database.add_document(
        "העברת כספים בין קופות גמל - תיקון",
        "url",
        "https://www.gov.il/example.docx",
        "source.txt",
        100,
    )
    doc = document("העברת כספים בין קופות גמל - תיקון", id=document_id)
    profile = build_document_profile(doc, TRANSFER_TEXT)
    integrity = assess_document_integrity(doc, TRANSFER_TEXT, profile)

    db = await database.get_db()
    try:
        await save_document_profile(db, profile, integrity)
        await save_document_profile(db, profile, integrity)
        row = await (
            await db.execute(
                "SELECT * FROM document_profiles WHERE document_id = ?",
                (document_id,),
            )
        ).fetchone()
        matches = await (
            await db.execute(
                "SELECT document_id FROM document_profiles_fts WHERE document_profiles_fts MATCH ?",
                ("העברת",),
            )
        ).fetchall()
    finally:
        await db.close()

    assert row["integrity_status"] == integrity["status"]
    assert json.loads(row["identity_evidence_json"])
    assert [match["document_id"] for match in matches] == [document_id]
