#!/usr/bin/env python3
"""Generate the D37 human-review sheet from the case, run, and groundedness artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cases = [
    json.loads(line)
    for line in (ROOT / "eval/d37_evidence_cases_pending_human_review.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
run = json.loads((ROOT / "results/d37_local_evidence_case_run.json").read_text(encoding="utf-8"))
grounded = json.loads((ROOT / "results/d37_groundedness.json").read_text(encoding="utf-8"))

run_by_id = {record["case"]["id"]: record for record in run["records"]}
score_by_id = {result["case_id"]: result["groundedness_score"] for result in grounded["results"]}

lines = [
    "# D37 — גיליון בדיקה אנושית לששת מקרי המבחן",
    "",
    "> סמן כל מקרה: ✅ נכון / ❌ שגוי / ⚠️ צריך שיפוט. ציון groundedness הוא signal, לא פסק דין.",
    "",
]

for index, case in enumerate(cases, 1):
    case_id = case["id"]
    record = run_by_id[case_id]
    score = score_by_id.get(case_id)
    lines += [
        f"## {index}. {case['question']}",
        "",
        f"- **expected disposition:** `{case['expected_disposition']}`",
        f"- **required claims:** {'; '.join(case['required_claims'])}",
        f"- **forbidden claims:** {'; '.join(case['forbidden_claims'])}",
        f"- **gate:** `{record['gate']['status']}` — **groundedness:** {score}",
        "",
        "**תשובת המודל:**",
        "",
        record["answer"],
        "",
        "**פסק דין אנושי:** [ ] ✅  [ ] ❌  [ ] ⚠️ — הערה: ",
        "",
    ]

output = ROOT / "results/d37_human_review_sheet.md"
output.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {output}")
