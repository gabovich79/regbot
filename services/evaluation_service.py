"""Small, deterministic helpers for RegBot retrieval evaluation."""

from services.metrics import average_precision, precision_at_k, recall_at_k, reciprocal_rank


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


def _normalize(value: str) -> str:
    return " ".join((value or "").split()).lower()


def _matches(gold: str, title: str, source_ref: str) -> bool:
    """True when a gold label matches a ranked document by title or source ref."""
    gold_n = _normalize(gold)
    if not gold_n:
        return False
    title_n = _normalize(title)
    ref_n = _normalize(source_ref)
    return gold_n == title_n or gold_n == ref_n or gold_n in title_n


def score_retrieval_ranking(case: dict, ranked_documents: list[dict], k: int = 20) -> dict:
    """Compute ranked IR metrics for one query against graded relevance labels.

    ``relevant_documents`` (or legacy ``expected_documents``) mark documents that
    must surface; ``distractor_documents`` mark documents that must not. A gold
    label matches a ranked document when it equals its title or source ref, or
    is a substring of its title (the corpus titles drift between short and long
    forms).
    """
    relevant = case.get("relevant_documents", case.get("expected_documents", []))
    distractors = case.get("distractor_documents", [])

    def label_for(doc: dict) -> str | None:
        title = doc.get("title") or ""
        ref = doc.get("source_ref") or ""
        for gold in relevant:
            if _matches(gold, title, ref):
                return gold
        for gold in distractors:
            if _matches(gold, title, ref):
                return f"distractor:{gold}"
        return None

    top_docs = ranked_documents[:k]
    ranked_items: list[str] = []
    for index, doc in enumerate(top_docs):
        label = label_for(doc)
        ranked_items.append(label if label else f"non:{index}")

    relevant_set = set(relevant)
    distractors_retrieved = [gold for gold in distractors if f"distractor:{gold}" in ranked_items]
    top_items = set(ranked_items)
    found = [gold for gold in relevant if gold in top_items]
    missing = [gold for gold in relevant if gold not in top_items]
    return {
        "id": case["id"],
        "recall_at_k": recall_at_k(ranked_items, relevant_set, k=k),
        "precision_at_k": precision_at_k(ranked_items, relevant_set, k=k),
        "mrr": reciprocal_rank(ranked_items, relevant_set),
        "average_precision": average_precision(ranked_items, relevant_set),
        "found_documents": found,
        "missing_documents": missing,
        "distractors_retrieved": distractors_retrieved,
        "passed": not missing and not distractors_retrieved,
    }
