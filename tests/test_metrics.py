import pytest

from services.metrics import (
    average_precision,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4, 9}  # 9 never retrieved
    assert recall_at_k(ranked, relevant, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevant, k=5) == pytest.approx(2 / 3)


def test_precision_at_k():
    ranked = [1, 2, 3, 4, 5]
    relevant = {2, 4}
    assert precision_at_k(ranked, relevant, k=3) == pytest.approx(1 / 3)
    assert precision_at_k(ranked, relevant, k=5) == pytest.approx(2 / 5)


def test_reciprocal_rank():
    assert reciprocal_rank([1, 2, 3], {3}) == pytest.approx(1 / 3)
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0
    assert reciprocal_rank([1, 2, 3], {99}) == 0.0


def test_average_precision():
    ranked = [1, 2, 3, 4]
    relevant = {2, 4}
    # precision@2 = 1/2, precision@4 = 2/4 -> AP = (0.5 + 0.5) / 2
    assert average_precision(ranked, relevant) == pytest.approx(0.5)


def test_metrics_handle_no_relevant_documents():
    ranked = [1, 2, 3]
    assert recall_at_k(ranked, set(), k=3) == 0.0
    assert reciprocal_rank(ranked, set()) == 0.0
    assert average_precision(ranked, set()) == 0.0
