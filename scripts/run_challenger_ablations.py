#!/usr/bin/env python3
"""Run bounded, reproducible candidate-fusion ablations on the challenger."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_challenger_hybrid import (
    CACHE,
    CASES,
    _embed_queries,
    evaluate_hybrid,
    section_index_from_nodes,
)

OUTPUT = Path("results/challenger_ablation_results.json")

DEFAULT_CONFIGS = [
    {
        "id": "union-5-node-1-catalog-0",
        "candidate_depth": 5,
        "node_rrf_weight": 1.0,
        "catalog_rrf_weight": 0.0,
    },
    {
        "id": "union-5-node-1-catalog-1",
        "candidate_depth": 5,
        "node_rrf_weight": 1.0,
        "catalog_rrf_weight": 1.0,
    },
    {
        "id": "union-5-node-1-catalog-2",
        "candidate_depth": 5,
        "node_rrf_weight": 1.0,
        "catalog_rrf_weight": 2.0,
    },
    {
        "id": "union-8-node-1-catalog-1",
        "candidate_depth": 8,
        "node_rrf_weight": 1.0,
        "catalog_rrf_weight": 1.0,
    },
]


async def run_ablations(
    profiles: list[dict],
    cases: list[dict],
    embed_queries: Callable[[list[str]], Awaitable[list[list[float]]]],
    *,
    configs: list[dict] | None = None,
    document_sections: dict[int, list[str]] | None = None,
    nodes: list[dict] | None = None,
) -> dict:
    """Evaluate named fusion configs using exactly one embedding call per question."""
    configs = configs or DEFAULT_CONFIGS
    questions = list(
        dict.fromkeys(
            case["question"] for case in cases if case.get("required_document_ids")
        )
    )
    vectors = await embed_queries(questions)
    if len(vectors) != len(questions):
        raise ValueError("Embedding provider returned an unexpected query-vector count")
    vector_by_question = dict(zip(questions, vectors))

    async def cached_queries(requested: list[str]) -> list[list[float]]:
        return [vector_by_question[question] for question in requested]

    runs = []
    for config in configs:
        result = await evaluate_hybrid(
            profiles,
            cases,
            cached_queries,
            document_sections=document_sections,
            nodes=nodes,
            candidate_depth=int(config["candidate_depth"]),
            node_rrf_weight=float(config["node_rrf_weight"]),
            catalog_rrf_weight=float(config.get("catalog_rrf_weight", 0.0)),
        )
        runs.append({"configuration": config, **result})
    return {"runs": runs}


def main() -> int:
    if not CACHE.exists():
        print(json.dumps({"error": f"cache missing: {CACHE}"}))
        return 1

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = asyncio.run(
        run_ablations(
            cache["profiles"],
            cases,
            _embed_queries,
            document_sections=section_index_from_nodes(cache["nodes"]),
            nodes=cache["nodes"],
        )
    )
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    "id": run["configuration"]["id"],
                    **run["metrics"],
                }
                for run in result["runs"]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
