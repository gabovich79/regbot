from scripts.build_ingestion_review_sheet import apply_duplicate_source_review


def test_duplicate_source_checksum_requires_human_review_for_each_duplicate():
    rows = [
        {
            "document_id": 16,
            "action_status": "ready_for_reingest",
            "source_checksum": "same",
            "integrity_reasons": [],
            "validation_errors": [],
        },
        {
            "document_id": 17,
            "action_status": "ready_for_reingest",
            "source_checksum": "same",
            "integrity_reasons": [],
            "validation_errors": [],
        },
        {
            "document_id": 18,
            "action_status": "ready_for_reingest",
            "source_checksum": "different",
            "integrity_reasons": [],
            "validation_errors": [],
        },
    ]

    apply_duplicate_source_review(rows)

    assert rows[0]["action_status"] == "needs_human_review"
    assert rows[1]["action_status"] == "needs_human_review"
    assert "duplicate_source_checksum" in rows[0]["integrity_reasons"]
    assert "duplicate_source_checksum" not in rows[2]["integrity_reasons"]
