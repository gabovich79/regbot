from services.evaluation_service import score_answer_response


def test_answer_score_requires_citations_and_required_terms():
    case = {
        "id": "transfer",
        "expected_citation_prefixes": ["D24-", "D20-"],
        "must_include": ["העברת כספים", "מידע"],
        "must_not_include": ["אין מקור"],
    }
    answer = "העברת כספים מחייבת מסירת מידע בין הגופים [D24-P2, D20-P4]."

    result = score_answer_response(case, answer)

    assert result["passed"] is True
    assert result["missing_citation_prefixes"] == []
    assert result["missing_required_terms"] == []


def test_answer_score_reports_missing_citation_and_prohibited_claim():
    case = {
        "id": "transfer",
        "expected_citation_prefixes": ["D24-"],
        "must_include": ["העברת כספים"],
        "must_not_include": ["פטור מלא ממס"],
    }

    result = score_answer_response(case, "העברת כספים פטור מלא ממס.")

    assert result["passed"] is False
    assert result["missing_citation_prefixes"] == ["D24-"]
    assert result["prohibited_terms_found"] == ["פטור מלא ממס"]


def test_answer_score_accepts_one_of_current_or_legacy_source_prefixes():
    case = {
        "id": "annual-cost",
        "any_citation_prefixes": ["D25-", "D34-"],
        "must_include": ["עלות"],
        "must_not_include": [],
    }

    result = score_answer_response(case, "העלות השנתית הצפויה מפורטת במסמך [D34-C3].")
    missing = score_answer_response(case, "העלות השנתית הצפויה מפורטת במסמך.")

    assert result["passed"] is True
    assert missing["passed"] is False
    assert missing["missing_any_citation_prefixes"] == ["D25-", "D34-"]
