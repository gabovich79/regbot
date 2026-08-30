#!/usr/bin/env python3
"""Hybrid dense+lexical document-retrieval evaluation for the challenger."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import numpy as np

from services.document_retriever import DocumentRetriever, _terms

CACHE = Path(
    os.environ.get(
        "CHALLENGER_CACHE_PATH",
        str(Path("results/challenger_embeddings_cache.json")),
    )
)
CASES = Path("eval/hierarchical_cases.jsonl")
RRF_K = 60


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denominator) if denominator else 0.0


def node_document_ranks(question: str, nodes: list[dict], document_ids: set[int]) -> dict[int, int]:
    """Rank documents by their single strongest raw legal node."""
    query_terms = _terms(question)
    best_scores: dict[int, float] = {}
    for node in nodes:
        document_id = int(node["document_id"])
        if document_id not in document_ids:
            continue
        heading_terms = _terms(str(node.get("heading") or ""))
        raw_terms = _terms(str(node.get("raw_text") or ""))
        score = 2 * len(query_terms & heading_terms) + len(query_terms & raw_terms)
        if score:
            best_scores[document_id] = max(best_scores.get(document_id, 0.0), score)
    ranked_ids = sorted(
        best_scores,
        key=lambda document_id: (best_scores[document_id], document_id),
        reverse=True,
    )
    return {document_id: rank for rank, document_id in enumerate(ranked_ids, 1)}


async def evaluate_hybrid(
    profiles: list[dict],
    cases: list[dict],
    embed_queries: Callable[[list[str]], Awaitable[list[list[float]]]],
    *,
    document_sections: dict[int, list[str]] | None = None,
    nodes: list[dict] | None = None,
    top_k: int = 5,
) -> dict:
    """Evaluate every unique question once and require full multi-source recall."""
    scored_cases = [case for case in cases if case.get("required_document_ids")]
    unique_questions = list(dict.fromkeys(case["question"] for case in scored_cases))
    vectors = await embed_queries(unique_questions)
    if len(vectors) != len(unique_questions):
        raise ValueError("Embedding provider returned an unexpected query-vector count")
    query_vectors = dict(zip(unique_questions, vectors))

    lexical = DocumentRetriever(profiles, document_sections=document_sections or {})
    rows = []
    for case in scored_cases:
        question = case["question"]
        required = set(case["required_document_ids"])
        lexical_results = lexical.retrieve(question, top_k=len(profiles))
        lexical_rank = {
            int(document["document_id"]): rank
            for rank, document in enumerate(lexical_results, 1)
        }
        dense_results = sorted(
            ((cosine(profile["embedding"], query_vectors[question]), profile) for profile in profiles),
            key=lambda item: item[0],
            reverse=True,
        )
        dense_rank = {
            int(profile["document_id"]): rank
            for rank, (_, profile) in enumerate(dense_results, 1)
        }
        all_document_ids = {int(profile["document_id"]) for profile in profiles}
        node_rank = node_document_ranks(question, nodes or [], all_document_ids)
        fused = sorted(
            (
                (
                    1 / (RRF_K + lexical_rank[int(profile["document_id"])])
                    + 1 / (RRF_K + dense_rank[int(profile["document_id"])])
                    + (
                        2 / (RRF_K + node_rank[int(profile["document_id"])])
                        if int(profile["document_id"]) in node_rank
                        else 0
                    ),
                    int(profile["document_id"]),
                )
                for profile in profiles
            ),
            reverse=True,
        )
        selected = [document_id for _, document_id in fused[:top_k]]
        retrieved_required = required & set(selected)
        first_hit = next(
            (rank for rank, document_id in enumerate(selected, 1) if document_id in required),
            None,
        )
        rows.append(
            {
                "id": case["id"],
                "required": sorted(required),
                "selected": selected,
                "hit_at": first_hit,
                "required_recall_at_3": round(
                    len(required & set(selected[:3])) / len(required), 3
                ),
                "required_recall_at_5": round(len(retrieved_required) / len(required), 3),
                "all_required_documents_at_3": int(required <= set(selected[:3])),
                "all_required_documents_at_5": int(required <= set(selected)),
                "diagnostics": {
                    "lexical_top_5": [
                        int(document["document_id"])
                        for document in lexical_results[:5]
                    ],
                    "dense_top_5": [
                        int(profile["document_id"])
                        for _, profile in dense_results[:5]
                    ],
                    "node_lexical_top_5": [
                        document_id
                        for document_id, _ in sorted(
                            node_rank.items(), key=lambda item: item[1]
                        )[:5]
                    ],
                    "fused_top_5": selected,
                },
            }
        )

    metrics = {
        "cases": len(rows),
        "document_recall_at_3": round(
            sum(row["required_recall_at_3"] for row in rows) / max(len(rows), 1), 3
        ),
        "document_recall_at_5": round(
            sum(row["required_recall_at_5"] for row in rows) / max(len(rows), 1), 3
        ),
        "all_required_documents_recall_at_3": round(
            sum(row["all_required_documents_at_3"] for row in rows) / max(len(rows), 1), 3
        ),
        "all_required_documents_recall_at_5": round(
            sum(row["all_required_documents_at_5"] for row in rows) / max(len(rows), 1), 3
        ),
    }
    return {"metrics": metrics, "rows": rows}


async def _embed_queries(questions: list[str]) -> list[list[float]]:
    from services.embeddings import embed_texts

    return await embed_texts(questions)


def section_index_from_nodes(nodes: list[dict]) -> dict[int, list[str]]:
    """Project cached hierarchical nodes into document-level lexical evidence."""
    index: dict[int, list[str]] = {}
    for node in nodes:
        document_id = int(node["document_id"])
        values = index.setdefault(document_id, [])
        heading = str(node.get("heading") or "").strip()
        raw_text = str(node.get("raw_text") or "").strip()
        if heading:
            values.append(heading)
        if raw_text:
            values.append(raw_text)
    return index


def main() -> int:
    if not CACHE.exists():
        print(json.dumps({"error": "cache missing; run build_challenger_embeddings.py first"}))
        return 1
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    profiles = cache["profiles"]
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = asyncio.run(
        evaluate_hybrid(
            profiles,
            cases,
            _embed_queries,
            document_sections=section_index_from_nodes(cache["nodes"]),
            nodes=cache["nodes"],
        )
    )
    output = Path("results/challenger_hybrid_metrics.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
