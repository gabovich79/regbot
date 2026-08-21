import pytest

from services.evaluation_service import score_retrieval_ranking


def _doc(title, source_ref=""):
    return {"title": title, "source_ref": source_ref}


def test_score_retrieval_ranking_computes_metrics_and_flags_distractors():
    case = {
        "id": "case-x",
        "relevant_documents": ["חוזר א", "חוזר ב"],
        "distractor_documents": ["חוזר ג"],
    }
    ranked = [_doc("חוזר א"), _doc("חוזר ג"), _doc("חוזר ב")]
    result = score_retrieval_ranking(case, ranked, k=3)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["distractors_retrieved"] == ["חוזר ג"]
    assert result["passed"] is False


def test_score_retrieval_ranking_supports_legacy_expected_documents():
    case = {"id": "case-y", "expected_documents": ["חוזר א"]}
    ranked = [_doc("חוזר ב"), _doc("חוזר א")]
    result = score_retrieval_ranking(case, ranked, k=2)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(0.5)
    assert result["passed"] is True


def test_score_retrieval_ranking_matches_short_gold_to_long_title():
    # Corpus titles drift between short and long forms.
    case = {"id": "case-drift", "relevant_documents": ["חוזר גמל 2020-9-2"]}
    ranked = [_doc("חוזר גמל 2020-9-2 העברת כספים")]
    result = score_retrieval_ranking(case, ranked, k=1)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["passed"] is True


def test_score_retrieval_ranking_matches_by_source_ref():
    case = {"id": "case-ref", "relevant_documents": ["https://example.test/circular.pdf"]}
    ranked = [_doc("כותרת משתנה", source_ref="https://example.test/circular.pdf")]
    result = score_retrieval_ranking(case, ranked, k=1)
    assert result["recall_at_k"] == pytest.approx(1.0)
