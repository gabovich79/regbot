"""Compare two web-answer benchmark runs and print a delta report.

Both inputs are result files produced by ``run_web_answer_benchmark.py`` (one
JSON object per line, each carrying the ``score`` dict from
``score_professional_answer``). No API keys or re-scoring required — this reads
the scores already recorded in each run.

Typical use — isolate "is it the model or the corpus?" by running the same 36
cases under two generation models and diffing:

    python scripts/compare_benchmark_runs.py \
      eval/results/web_answer_results.jsonl \
      eval/results/web_answer_results_pro.jsonl \
      --labels flash pro

Exit code is 0 when the candidate's strict pass count is >= the baseline's,
1 otherwise, so it can gate a CI/experiment step.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCORE_FIELDS = [
    ("conclusion_score", "conclusion"),
    ("required_concepts_score", "concepts"),
    ("actionability_score", "action"),
    ("clarification_score", "clarify"),
]


def load_run(path: Path) -> dict[str, dict]:
    """Index one result file by case id."""
    runs: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        runs[record["id"]] = record
    return runs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _passed(record: dict) -> bool:
    return bool(record.get("score", {}).get("passed"))


def _errored(record: dict) -> bool:
    return record.get("error") is not None or "error" in record.get("score", {})


def _aggregate(records: list[dict]) -> dict:
    """Pass count, error count, and mean sub-scores for a set of records."""
    stats = {
        "n": len(records),
        "passed": sum(1 for r in records if _passed(r)),
        "errors": sum(1 for r in records if _errored(r)),
    }
    for field, label in SCORE_FIELDS:
        stats[label] = _mean([r.get("score", {}).get(field, 0.0) for r in records])
    return stats


def _fmt_delta(base: float, cand: float, *, as_int: bool = False) -> str:
    delta = cand - base
    if as_int:
        base_s, cand_s, delta_s = f"{int(base)}", f"{int(cand)}", f"{delta:+d}" if delta else "  0"
        return f"{base_s:>4} -> {cand_s:>4} ({delta_s})"
    arrow = "+" if delta > 1e-9 else ("-" if delta < -1e-9 else " ")
    return f"{base:.3f} -> {cand:.3f} ({arrow}{abs(delta):.3f})"


def _print_block(title: str, base: dict, cand: dict) -> None:
    print(f"\n{title}  (n={base['n']})")
    print(f"  strict pass   {_fmt_delta(base['passed'], cand['passed'], as_int=True)}")
    if base["errors"] or cand["errors"]:
        print(f"  errors        {_fmt_delta(base['errors'], cand['errors'], as_int=True)}")
    for _, label in SCORE_FIELDS:
        print(f"  {label:<12}  {_fmt_delta(base[label], cand[label])}")


def compare(baseline_path: Path, candidate_path: Path, labels: tuple[str, str]) -> int:
    base_run = load_run(baseline_path)
    cand_run = load_run(candidate_path)

    base_ids, cand_ids = set(base_run), set(cand_run)
    shared = base_ids & cand_ids
    if not shared:
        raise SystemExit("No shared case ids between the two runs — cannot compare.")
    only_base = base_ids - cand_ids
    only_cand = cand_ids - base_ids
    if only_base:
        print(f"[warn] {len(only_base)} case(s) only in {labels[0]}: {sorted(only_base)}")
    if only_cand:
        print(f"[warn] {len(only_cand)} case(s) only in {labels[1]}: {sorted(only_cand)}")

    base_records = [base_run[i] for i in shared]
    cand_records = [cand_run[i] for i in shared]

    print("=" * 60)
    print(f"Benchmark comparison: {labels[0]} (baseline) -> {labels[1]} (candidate)")
    print(f"Baseline : {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print("=" * 60)

    _print_block("OVERALL", _aggregate(base_records), _aggregate(cand_records))

    # Per-category
    by_cat_base: dict[str, list[dict]] = defaultdict(list)
    by_cat_cand: dict[str, list[dict]] = defaultdict(list)
    for case_id in shared:
        category = base_run[case_id].get("category") or "uncategorized"
        by_cat_base[category].append(base_run[case_id])
        by_cat_cand[category].append(cand_run[case_id])

    print("\n" + "-" * 60)
    print("PER CATEGORY")
    for category in sorted(by_cat_base, key=lambda c: -len(by_cat_base[c])):
        _print_block(category, _aggregate(by_cat_base[category]), _aggregate(by_cat_cand[category]))

    # Per-case flips
    gained = sorted(i for i in shared if not _passed(base_run[i]) and _passed(cand_run[i]))
    lost = sorted(i for i in shared if _passed(base_run[i]) and not _passed(cand_run[i]))
    print("\n" + "-" * 60)
    print(f"FLIPS   FAIL->PASS: {len(gained)}   PASS->FAIL: {len(lost)}")
    for case_id in gained:
        print(f"  + {case_id}")
    for case_id in lost:
        print(f"  - {case_id}")

    base_pass = sum(1 for r in base_records if _passed(r))
    cand_pass = sum(1 for r in cand_records if _passed(r))
    print("\n" + "=" * 60)
    verdict = "candidate >= baseline" if cand_pass >= base_pass else "candidate REGRESSED"
    print(f"Verdict: {verdict}  ({base_pass} -> {cand_pass} strict passes)")
    return 0 if cand_pass >= base_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files")
    parser.add_argument("baseline", type=Path, help="baseline result jsonl (e.g. flash run)")
    parser.add_argument("candidate", type=Path, help="candidate result jsonl (e.g. pro run)")
    parser.add_argument(
        "--labels", nargs=2, metavar=("BASE", "CAND"), default=("baseline", "candidate"),
        help="short names for the two runs, used in the report headers",
    )
    args = parser.parse_args()
    raise SystemExit(compare(args.baseline, args.candidate, tuple(args.labels)))


if __name__ == "__main__":
    main()
