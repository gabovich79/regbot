#!/usr/bin/env python3
"""Run the full challenger promotion gate: tuning + held-out + legacy cases."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.corpus_scope import active_cases, blocked_cases, load_active_document_ids, load_cases
from scripts.measure_challenger_hybrid import (
    CACHE,
    _embed_queries,
    evaluate_hybrid,
    section_index_from_nodes,
)

TUNING_CASES = Path("eval/hierarchical_cases.jsonl")
HELDOUT_CASES = Path("eval/hierarchical_cases_heldout.jsonl")
LEGACY_CASES = Path("eval/retrieval_cases.jsonl")
LEGACY_MAP = Path("eval/legacy_case_document_map.json")
OUTPUT = Path("results/challenger_promotion_gate.json")

WINNER = {"candidate_depth": 5, "node_rrf_weight": 1.0, "catalog_rrf_weight": 1.0}
PROMOTION_TUNING_REVIEW_STATUS = "human_verified"
MIN_TUNING_CASES_FOR_TOP3 = 7


def split_promotion_tuning_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate verified promotion evidence from provisional tuning hypotheses."""
    accepted = [
        case for case in cases
        if case.get("review_status") == PROMOTION_TUNING_REVIEW_STATUS
    ]
    excluded = [
        {
            "id": case["id"],
            "reason": f"review_status:{case.get('review_status', 'missing')}",
        }
        for case in cases
        if case.get("review_status") != PROMOTION_TUNING_REVIEW_STATUS
    ]
    return accepted, excluded


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_legacy_cases(active_ids: set[str] | None = None) -> list[dict]:
    mapping = json.loads(LEGACY_MAP.read_text(encoding="utf-8"))
    cases = []
    for case in load_jsonl(LEGACY_CASES):
        required = mapping.get(case["id"])
        if not required:
            continue
        scoped = {
            "id": case["id"],
            "question": case["question"],
            "required_document_ids": required,
        }
        if active_ids is not None:
            from scripts.corpus_scope import annotate_case_blocked

            annotate_case_blocked(scoped, active_ids)
        cases.append(scoped)
    return cases


def promotion_decision(
    metrics: dict,
    failed: dict[str, list[str]],
    *,
    enforce_tuning_top3: bool = True,
    sufficient_verified_tuning_cases: bool = True,
) -> bool:
    tuning = metrics["tuning"]
    return (
        sufficient_verified_tuning_cases
        and tuning["all_required_documents_recall_at_5"] == 1.0
        and (
            not enforce_tuning_top3
            or tuning["all_required_documents_recall_at_3"] >= 0.857
        )
        and not failed["heldout"]
        and not failed["legacy"]
    )


async def run() -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    profiles = cache["profiles"]
    nodes = cache["nodes"]
    document_sections = section_index_from_nodes(nodes)

    # Embed all questions across all three groups in one batch.
    active_ids = load_active_document_ids()
    groups = {
        "tuning": load_cases(TUNING_CASES, active_ids),
        "heldout": load_cases(HELDOUT_CASES, active_ids),
        "legacy": load_legacy_cases(active_ids),
    }
    groups["tuning"], excluded_tuning_cases = split_promotion_tuning_cases(
        groups["tuning"]
    )
    sufficient_verified_tuning_cases = (
        len(groups["tuning"]) >= MIN_TUNING_CASES_FOR_TOP3
    )
    enforce_tuning_top3 = sufficient_verified_tuning_cases
    all_questions = list(
        dict.fromkeys(
            question
            for cases in groups.values()
            for question in (
                case["question"] for case in cases if case.get("required_document_ids")
            )
        )
    )
    vectors = await _embed_queries(all_questions)
    vector_by_question = dict(zip(all_questions, vectors))

    async def cached_queries(questions: list[str]) -> list[list[float]]:
        return [vector_by_question[question] for question in questions]

    results = {}
    for group_name, cases in groups.items():
        scoped_cases = active_cases(cases)
        result = await evaluate_hybrid(
            profiles,
            scoped_cases,
            cached_queries,
            document_sections=document_sections,
            nodes=nodes,
            candidate_depth=WINNER["candidate_depth"],
            node_rrf_weight=WINNER["node_rrf_weight"],
            catalog_rrf_weight=WINNER["catalog_rrf_weight"],
        )
        result["blocked_cases"] = [
            {"id": case["id"], "reason": case.get("blocked_reason")}
            for case in blocked_cases(cases)
        ]
        results[group_name] = result

    metrics = {name: results[name]["metrics"] for name in results}
    failed = {
        name: [
            row["id"]
            for row in results[name]["rows"]
            if not row["all_required_documents_at_5"]
        ]
        for name in results
    }

    tuning_all5 = metrics["tuning"]["all_required_documents_recall_at_5"]
    tuning_all3 = metrics["tuning"]["all_required_documents_recall_at_3"]
    promotion = promotion_decision(
        metrics,
        failed,
        enforce_tuning_top3=enforce_tuning_top3,
        sufficient_verified_tuning_cases=sufficient_verified_tuning_cases,
    )

    gate = {
        "winner_config": WINNER,
        "promotion_policy": {
            "tuning_review_status": PROMOTION_TUNING_REVIEW_STATUS,
            "tuning_case_count": len(groups["tuning"]),
            "sufficient_verified_tuning_cases": sufficient_verified_tuning_cases,
            "top3_enforced": enforce_tuning_top3,
            "top3_minimum_case_count": MIN_TUNING_CASES_FOR_TOP3,
        },
        "excluded_cases": {"tuning": excluded_tuning_cases},
        "metrics": metrics,
        "failed_all_required_at_5": failed,
        "rows": {
            name: [
                {
                    "id": row["id"],
                    "required": row["required"],
                    "selected": row["selected"],
                    "diagnostics": row.get("diagnostics", {}),
                }
                for row in results[name]["rows"]
            ]
            for name in results
        },
        "blocked_cases": {
            name: results[name].get("blocked_cases", [])
            for name in results
        },
        "promotion": promotion,
        "reasons": [] if promotion else [
            reason
            for reason in (
                (
                    "insufficient human-verified tuning cases: "
                    f"{len(groups['tuning'])}<{MIN_TUNING_CASES_FOR_TOP3}"
                    if not sufficient_verified_tuning_cases
                    else None
                ),
                "tuning all-required@5 below 1.0" if tuning_all5 < 1.0 else None,
                (
                    "tuning all-required@3 below 0.857"
                    if enforce_tuning_top3 and tuning_all3 < 0.857
                    else None
                ),
                f"heldout failures: {failed['heldout']}" if failed["heldout"] else None,
                f"legacy failures: {failed['legacy']}" if failed["legacy"] else None,
            )
            if reason
        ],
    }
    OUTPUT.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return gate


def main() -> int:
    if not CACHE.exists():
        print(json.dumps({"error": f"cache missing: {CACHE}"}))
        return 1
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
