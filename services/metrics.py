"""Pure information-retrieval metrics for RegBot's golden-set evaluation.

These functions operate on an ordered list of retrieved items (``ranked``) and
the set of relevant items (``relevant``). They hold no I/O and no state, so the
evaluation harness stays deterministic and unit-testable.
"""

from collections.abc import Iterable


def recall_at_k(ranked: list, relevant: Iterable, k: int) -> float:
    """Fraction of relevant items surfaced within the top-k results."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def precision_at_k(ranked: list, relevant: Iterable, k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    relevant = set(relevant)
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def reciprocal_rank(ranked: list, relevant: Iterable) -> float:
    """Reciprocal of the rank of the first relevant item (1-indexed)."""
    relevant = set(relevant)
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked: list, relevant: Iterable) -> float:
    """Average precision for a single query over a graded relevance set."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            score += hits / index
    return score / len(relevant)


def mean(metric_fn, ranked_list: list[list], relevant_list: list[Iterable], **kwargs) -> float:
    """Average a per-query metric across a list of queries."""
    if not ranked_list:
        return 0.0
    values = [metric_fn(ranked, relevant, **kwargs) for ranked, relevant in zip(ranked_list, relevant_list)]
    return sum(values) / len(values)
