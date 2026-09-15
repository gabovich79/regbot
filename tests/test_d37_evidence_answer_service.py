from services.d37_evidence_answer_service import build_grounded_prompt, gate_grounded_answer


HITS = [
    {
        "evidence_id": "D37-9-16A-A",
        "section": "9(16א)(א)",
        "page_start": 43,
        "page_end": 43,
        "verification_status": "machine_observed_pending_human_review",
        "raw_text": "תנאי משיכה מקרן השתלמות לאחר שש שנים",
    },
    {
        "evidence_id": "D37-125G-D-5",
        "section": "125ג(ד)(5)",
        "page_start": 204,
        "page_end": 204,
        "verification_status": "machine_observed_pending_human_review",
        "raw_text": "הפניה לסעיף 121 לגבי ריבית",
    },
]


def test_grounded_prompt_requires_pointer_only_not_text_transcription():
    prompt = build_grounded_prompt("מה המס?", HITS)

    assert "D37-9-16A-A" in prompt
    assert "D37-125G-D-5" in prompt
    assert "CONFIDENCE HIGH" in prompt
    assert "[[evidence_id]]" in prompt
    assert "אין להעתיק" in prompt
    assert "הקשר חיצוני" in prompt


def test_gate_passes_when_evidence_is_cited_by_bare_pointer():
    result = gate_grounded_answer(
        "המקור מפנה לסעיף 121. [[D37-125G-D-5]]",
        HITS,
    )

    assert result["status"] == "pass"
    assert result["citation_ids"] == ["D37-125G-D-5"]
    assert result["confidence"] == "not_high_pending_human_review"


def test_gate_rejects_unknown_citations_and_high_confidence_claims():
    result = gate_grounded_answer(
        "CONFIDENCE HIGH: תשובה. [[D37-invented]]",
        HITS,
    )

    assert result["status"] == "reject"
    assert result["unknown_citation_ids"] == ["D37-invented"]
    assert "high_confidence" in result["reasons"]


def test_gate_rejects_an_answer_with_no_citation():
    result = gate_grounded_answer("תשובה בלי אף מקור.", HITS)

    assert result["status"] == "reject"
    assert "missing_citation" in result["reasons"]
