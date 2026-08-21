"""Evaluate complete RegBot answers against deterministic source/term guardrails.

Run from Render Shell:
    python scripts/run_answer_evaluation.py

This invokes Gemini and OpenAI retrieval, but does not modify documents or
SQLite messages. It reports whether each answer contains expected evidence IDs
and domain terms.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.database import get_db, init_db
from services.claude_service import stream_chat
from services.evaluation_service import score_answer_response

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "eval" / "answer_cases.jsonl"


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def answer_for_case(case: dict, db) -> str:
    answer = ""
    async for event in stream_chat(case["question"], db, conversation_history=[]):
        if event["type"] == "text":
            answer += event["text"]
        elif event["type"] == "error":
            raise RuntimeError(event["text"])
    return answer


async def run(cases_path: Path, *, show_failures: bool):
    await init_db()
    cases = load_cases(cases_path)
    db = await get_db()
    try:
        results = []
        for case in cases:
            answer = await answer_for_case(case, db)
            result = score_answer_response(case, answer)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {case['id']}")
            if not result["passed"]:
                print(f"  missing citations: {result['missing_citation_prefixes']}")
                print(f"  missing alternative citations: {result['missing_any_citation_prefixes']}")
                print(f"  missing terms: {result['missing_required_terms']}")
                print(f"  prohibited terms: {result['prohibited_terms_found']}")
                if show_failures:
                    print("  answer:")
                    print(answer)

        passed = sum(result["passed"] for result in results)
        total = len(results)
        print(f"\nAnswer Evaluation: {passed}/{total} ({passed / total:.1%})")
        return 0 if passed == total else 1
    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete RegBot answers")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--show-failures", action="store_true",
        help="print full generated answers for failed cases",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.cases, show_failures=args.show_failures)))


if __name__ == "__main__":
    main()
