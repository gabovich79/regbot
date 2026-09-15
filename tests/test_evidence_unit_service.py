import hashlib

import pytest

from services.evidence_unit_service import build_evidence_units


def test_evidence_unit_extracts_a_contiguous_annotated_span_with_provenance():
    pages = [
        {"page_number": 43, "text": "TITLE משיכת תשלומי מעביד\nSTART סכומים שמשך עובד\nתנאי פטור\nEND משיכה לעצמאים"},
    ]
    units = build_evidence_units(
        pages,
        source_id="D37",
        source_checksum="a" * 64,
        annotations=[
            {
                "evidence_id": "D37-9-16A-A",
                "section": "9(16א)(א)",
                "page_start": 43,
                "page_end": 43,
                "start_anchor": "TITLE משיכת תשלומי מעביד",
                "end_exclusive_anchor": "END משיכה לעצמאים",
                "verification_status": "machine_observed_pending_human_review",
            }
        ],
    )

    unit = units[0]
    assert unit["section"] == "9(16א)(א)"
    assert unit["source_id"] == "D37"
    assert unit["page_start"] == 43
    assert unit["page_end"] == 43
    assert unit["source_checksum"] == "a" * 64
    assert unit["raw_text"] == "TITLE משיכת תשלומי מעביד\nSTART סכומים שמשך עובד\nתנאי פטור"
    assert unit["span_hash"] == hashlib.sha256(unit["raw_text"].encode("utf-8")).hexdigest()


def test_evidence_unit_rejects_an_annotation_when_its_anchor_is_not_on_the_declared_pages():
    with pytest.raises(ValueError, match="start_anchor"):
        build_evidence_units(
            [{"page_number": 43, "text": "source text"}],
            source_id="D37",
            source_checksum="a" * 64,
            annotations=[
                {
                    "evidence_id": "D37-missing",
                    "section": "9(16א)(א)",
                    "page_start": 43,
                    "page_end": 43,
                    "start_anchor": "not present",
                    "verification_status": "machine_observed_pending_human_review",
                }
            ],
        )
