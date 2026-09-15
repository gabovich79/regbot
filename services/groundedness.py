"""Local groundedness scoring: judge whether an answer is entailed by retrieved evidence.

Model-agnostic. A concrete scorer supplies ``predict_fn`` — for example the
entailment probability from a multilingual NLI model (XLM-R-large-XNLI), or a
"consistent"-label score from HHEM.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_premise(evidence_units: list[dict[str, Any]]) -> str:
    """Concatenate the verbatim retrieved evidence into a single premise."""
    return "\n\n".join(str(unit.get("raw_text", "")) for unit in evidence_units)


def score_answer_groundedness(
    answer: str,
    evidence_units: list[dict[str, Any]],
    predict_fn: Callable[[list[tuple[str, str]]], list[float]],
) -> dict[str, Any]:
    """Score how consistent an answer is with the verbatim retrieved evidence.

    ``predict_fn`` receives (premise, hypothesis) pairs and returns one score per
    pair in [0, 1] (0 = unsupported, 1 = fully supported). It is injected so this
    unit can be tested without loading a model.
    """
    premise = build_premise(evidence_units)
    scores = predict_fn([(premise, answer)])
    return {
        "groundedness_score": float(scores[0]),
        "premise": premise,
        "hypothesis": answer,
    }
