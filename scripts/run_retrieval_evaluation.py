"""Measure whether RegBot retrieval surfaces the expected documents.

Run from Render Shell after deploying:
    python scripts/run_retrieval_evaluation.py

The command sends embedding queries but does not call Gemini and does not modify
SQLite. It prints one result per case plus a Retrieval Recall summary.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.database import get_db, init_db
from services.evaluation_service import score_retrieval_context
from services.rag_service import retrieve_relevant_chunks

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
            context = await retrieve_relevant_chunks(
                case["question"], db, top_k=top_k, context_window=1
            )
            result = score_retrieval_context(case, context)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {case['id']}")
            if result["missing_documents"]:
                print(f"  missing: {', '.join(result['missing_documents'])}")

        passed = sum(result["passed"] for result in results)
        total = len(results)
        print(f"\nRetrieval Recall@{top_k}: {passed}/{total} ({passed / total:.1%})")
        return 0 if passed == total else 1
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
