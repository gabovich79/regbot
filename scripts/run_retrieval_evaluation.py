"""Measure whether RegBot retrieval surfaces the expected documents.

Run from Render Shell after deploying:
    python scripts/run_retrieval_evaluation.py

The command sends embedding queries but does not call Gemini and does not modify
SQLite. It prints per-case results plus ranked metrics (Recall@k, Precision@k,
MRR, MAP) against the golden set in ``eval/retrieval_cases.jsonl``.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.database import get_db, init_db
from services.evaluation_service import score_retrieval_ranking
from services.rag_service import retrieve_ranked_documents

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "eval" / "retrieval_cases.jsonl"


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def run(cases_path: Path, top_k: int):
    await init_db()
    cases = load_cases(cases_path)
    db = await get_db()
    try:
        results = []
        for case in cases:
            ranked = await retrieve_ranked_documents(case["question"], db, top_k=top_k)
            result = score_retrieval_ranking(case, ranked, k=top_k)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{status}] {case['id']}  "
                f"recall@{top_k}={result['recall_at_k']:.2f} "
                f"MRR={result['mrr']:.3f}"
            )
            if result["missing_documents"]:
                print(f"  missing: {', '.join(result['missing_documents'])}")
            if result["distractors_retrieved"]:
                print(f"  distractors retrieved: {', '.join(result['distractors_retrieved'])}")

        total = len(results)
        if total == 0:
            print("No cases found.")
            return 1

        def mean(key: str) -> float:
            return sum(result[key] for result in results) / total

        print(f"\n=== Retrieval evaluation ({total} cases, top_k={top_k}) ===")
        print(f"Recall@{top_k}:    {mean('recall_at_k'):.1%}")
        print(f"Precision@{top_k}: {mean('precision_at_k'):.1%}")
        print(f"MRR:              {mean('mrr'):.3f}")
        print(f"MAP:              {mean('average_precision'):.3f}")
        print(f"Cases passed:     {sum(result['passed'] for result in results)}/{total}")
        return 0 if all(result["passed"] for result in results) else 1
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RegBot retrieval against known cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.cases, args.top_k)))


if __name__ == "__main__":
    main()
