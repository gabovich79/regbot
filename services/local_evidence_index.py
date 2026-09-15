"""Small, local vector index for verified evidence units."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

Embedding = list[float]


def _validated_vector(value: Any, *, expected_dimension: int | None = None) -> Embedding:
    if not isinstance(value, list) or not value:
        raise ValueError("embedding must be a non-empty list")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("embedding must contain only finite numbers")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError("embedding dimension does not match the local evidence index")
    return vector


def _cosine(left: Embedding, right: Embedding) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding must not be a zero vector")
    return numerator / (left_norm * right_norm)


def build_local_evidence_index(
    evidence_units: list[dict[str, Any]],
    *,
    model: str,
    embed_many: Callable[[list[str]], list[Embedding]],
) -> dict[str, Any]:
    """Embed evidence units locally and return a JSON-serializable index."""
    if not evidence_units:
        raise ValueError("at least one evidence unit is required")
    texts = [str(unit.get("raw_text") or "") for unit in evidence_units]
    if any(not text for text in texts):
        raise ValueError("every evidence unit requires raw_text")

    embeddings = embed_many(texts)
    if len(embeddings) != len(evidence_units):
        raise ValueError("embed_many must return one embedding per evidence unit")

    vectors = [_validated_vector(vector) for vector in embeddings]
    dimension = len(vectors[0])
    for vector in vectors[1:]:
        _validated_vector(vector, expected_dimension=dimension)

    return {
        "schema_version": "local-evidence-index/v1",
        "model": model,
        "dimension": dimension,
        "entries": [
            {"evidence_id": unit["evidence_id"], "embedding": vector, "evidence_unit": unit}
            for unit, vector in zip(evidence_units, vectors)
        ],
    }


def search_local_evidence_index(
    question: str,
    index: dict[str, Any],
    *,
    embed_query: Callable[[str], Embedding],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return ranked evidence units from a local, same-model vector index."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    dimension = int(index["dimension"])
    query_vector = _validated_vector(embed_query(question), expected_dimension=dimension)
    ranked: list[dict[str, Any]] = []
    for entry in index.get("entries", []):
        vector = _validated_vector(entry["embedding"], expected_dimension=dimension)
        ranked.append({**entry["evidence_unit"], "score": _cosine(query_vector, vector)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
