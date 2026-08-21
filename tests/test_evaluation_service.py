from services.evaluation_service import score_retrieval_context


def test_retrieval_score_requires_all_expected_documents():
    context = "=== חוזר גמל 2024-9-8 ===\n...\n=== חוזר גמל 2020-9-2 העברת כספים ==="
    case = {
        "id": "complex-transfer",
        "expected_documents": ["חוזר גמל 2024-9-8", "חוזר גמל 2020-9-2"],
    }

    result = score_retrieval_context(case, context)

    assert result["passed"] is True
    assert result["found_documents"] == case["expected_documents"]


def test_retrieval_score_reports_missing_expected_document():
    case = {
        "id": "missing-document",
        "expected_documents": ["חוזר גמל 2024-9-8", "חוזר גמל 2020-9-2"],
    }

    result = score_retrieval_context(case, "=== חוזר גמל 2024-9-8 ===")

    assert result["passed"] is False
    assert result["missing_documents"] == ["חוזר גמל 2020-9-2"]
