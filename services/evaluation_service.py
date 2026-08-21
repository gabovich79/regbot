"""Small, deterministic helpers for RegBot retrieval evaluation."""


def score_retrieval_context(case: dict, context: str) -> dict:
    """Score whether all expected document titles appeared in retrieved context."""
    expected = case.get("expected_documents", [])
    found = [title for title in expected if title in context]
    missing = [title for title in expected if title not in context]
    return {
        "id": case["id"],
        "passed": not missing,
        "found_documents": found,
        "missing_documents": missing,
    }
