"""Run the web-derived professional-answer benchmark against a live RegBot API.

Usage:
    python scripts/run_web_answer_benchmark.py \
      --cases eval/web_answer_cases.jsonl \
      --output eval/results/web_answer_results.jsonl

The script sends only the question to the bot. Reference answers and source
URLs remain outside the prompt and are used only for scoring/reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.evaluation_service import score_professional_answer

DEFAULT_URL = "https://regbot-wly9.onrender.com/api/chat"


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ask_bot(url: str, question: str) -> str:
    payload = urllib.parse.urlencode({"question": question}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(request, timeout=180) as response:
        answer_parts = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "text":
                answer_parts.append(event.get("text", ""))
            elif event.get("type") == "error":
                raise RuntimeError(event.get("text", "unknown RegBot error"))
        return "".join(answer_parts)


def run(cases_path: Path, output_path: Path, url: str) -> int:
    cases = load_cases(cases_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with output_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            started = time.monotonic()
            try:
                answer = ask_bot(url, case["question"])
                score = score_professional_answer(case, answer)
                error = None
            except Exception as exc:
                answer = ""
                score = {"id": case["id"], "passed": False, "error": str(exc)}
                error = str(exc)
            result = {
                "id": case["id"],
                "category": case.get("category"),
                "question": case["question"],
                "reference_answer": case.get("reference_answer"),
                "reference_url": case.get("reference_url"),
                "answer": answer,
                "score": score,
                "error": error,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if score.get("passed"):
                passed += 1
            print(f"[{index}/{len(cases)}] {'PASS' if score.get('passed') else 'FAIL'} {case['id']}")

    print(f"\nProfessional Answer Benchmark: {passed}/{len(cases)} ({passed / len(cases):.1%})")
    print(f"Results: {output_path}")
    return 0 if passed == len(cases) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run web-derived professional-answer benchmark")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    raise SystemExit(run(args.cases, args.output, args.url))


if __name__ == "__main__":
    main()
