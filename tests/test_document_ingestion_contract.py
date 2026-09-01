from services.document_ingestion_service import build_ingestion_receipt


TRANSFER_TEXT = """
חוזר גופים מוסדיים 2016-9-11
העברת כספים בין קופות גמל - תיקון
כללי
הגדרות
טיפול בבקשת העברה
הגוף המנהל יעביר את הכספים בהתאם לבקשת העמית ובמועדים הקבועים בהוראות אלה.
"""


def test_receipt_for_valid_document_contains_source_derived_artifacts():
    document = {
        "id": 22,
        "title": "העברת כספים בין קופות גמל - תיקון",
        "source_type": "docx",
        "source_ref": "original.docx",
        "effective_date": None,
        "valid_until": None,
        "superseded_by": None,
        "topic": "ניודים, העברות בין קופות",
        "document_type": "חוזר",
        "lifecycle_status": "current",
    }

    receipt = build_ingestion_receipt(
        document,
        TRANSFER_TEXT,
        original_path="/sources/22.docx",
        source_checksum="a" * 64,
    )

    assert receipt["status"] == "validated"
    assert receipt["document_id"] == 22
    assert receipt["source"]["checksum"] == "a" * 64
    assert receipt["profile"]["official_number"] == "2016-9-11"
    assert receipt["profile"]["identity_evidence"]
    assert receipt["integrity"]["status"] == "verified"
    assert receipt["counts"]["text_characters"] > 0
    assert receipt["counts"]["meaningful_nodes"] >= 1
    assert "העברת" in receipt["keywords"]
    assert receipt["validation_errors"] == []


def test_receipt_rejects_empty_extraction_as_needs_reupload():
    receipt = build_ingestion_receipt(
        {"id": 99, "title": "מסמך בדיקה", "source_type": "pdf"},
        "",
        original_path="/sources/99.pdf",
        source_checksum="b" * 64,
    )

    assert receipt["status"] == "needs_reupload"
    assert "empty_extracted_text" in receipt["validation_errors"]
