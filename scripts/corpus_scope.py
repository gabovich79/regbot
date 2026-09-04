"""Align evaluation cases with the approved active corpus."""

from __future__ import annotations

import json
from pathlib import Path

DECISIONS_PATH = Path("eval/corpus_decisions.json")
MANIFEST_PATH = Path("eval/production_corpus_manifest_2026-08-29.json")


def load_active_document_ids() -> set[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))
    inactive = (
        set(decisions.get("excluded_from_corpus", {}))
        | set(decisions.get("duplicates", {}))
        | set(decisions.get("unresolved_source", {}))
        | set(decisions.get("metadata_review_pending", {}))
    )
    return {str(doc["id"]) for doc in manifest if str(doc["id"]) not in inactive}


def annotate_case_blocked(case: dict, active_ids: set[str]) -> dict:
    """Return case with blocked reason if it requires an inactive document."""
    required = [str(x) for x in case.get("required_document_ids", [])]
    missing = [doc_id for doc_id in required if doc_id not in active_ids]
    if missing:
        case["blocked_reason"] = f"requires_inactive_documents:{','.join(sorted(missing))}"
    return case


def load_cases(path: Path, active_ids: set[str]) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        annotate_case_blocked(case, active_ids)
        cases.append(case)
    return cases


def active_cases(cases: list[dict]) -> list[dict]:
    return [case for case in cases if "blocked_reason" not in case]


def blocked_cases(cases: list[dict]) -> list[dict]:
    return [case for case in cases if "blocked_reason" in case]
