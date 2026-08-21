"""Small, deterministic helpers for RegBot retrieval evaluation."""


def score_answer_response(case: dict, answer: str) -> dict:
    """Apply deterministic guardrails to a generated regulatory answer."""
    expected_prefixes = case.get("expected_citation_prefixes", [])
    any_prefixes = case.get("any_citation_prefixes", [])
    required_terms = case.get("must_include", [])
    prohibited_terms = case.get("must_not_include", [])
    missing_citations = [prefix for prefix in expected_prefixes if prefix not in answer]
    missing_any_citations = any_prefixes if any_prefixes and not any(prefix in answer for prefix in any_prefixes) else []
    missing_terms = [term for term in required_terms if term not in answer]
    prohibited_found = [term for term in prohibited_terms if term in answer]
    return {
        "id": case["id"],
        "passed": not (missing_citations or missing_any_citations or missing_terms or prohibited_found),
        "missing_citation_prefixes": missing_citations,
        "missing_any_citation_prefixes": missing_any_citations,
        "missing_required_terms": missing_terms,
        "prohibited_terms_found": prohibited_found,
    }


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
