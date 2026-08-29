#!/usr/bin/env python3
"""Compute retrieval metrics for the hierarchical challenger."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOCUMENTS_DIR
from services.document_retriever import DocumentRetriever
from services.document_profile_service import build_document_profile
from services.legal_parser import build_legal_tree

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
TEXT_ROOT = Path("/tmp/regbot-hierarchical-corpus/texts")
if Path(DOCUMENTS_DIR).exists():
    TEXT_ROOT = Path(DOCUMENTS_DIR)
CASES = Path("eval/hierarchical_cases.jsonl")


def build_document_sections() -> dict[int, list[str]]:
    import re as _re

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sections: dict[int, list[str]] = {}
    for doc in manifest:
        text = (TEXT_ROOT / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
        paragraphs = [line for line in text.splitlines() if line.strip()]
        tree = build_legal_tree(paragraphs, doc)
        stack = [tree]
        labels = []
        while stack:
            node = stack.pop()
            if node["node_type"] in {"section", "subsection", "chapter"}:
                labels.append(node["heading"])
            stack.extend(reversed(node.get("children", [])))
        # Include every explicit "סעיף N" mention even when it is inline text.
        # RTL extraction inserts whitespace inside numbers ("סעיף 8 (ד)"), so
        # search a whitespace-normalized copy as well.
        for match in _re.finditer(r"סעיף\s*\d+(?:\([^)]*\))?", text):
            labels.append(match.group(0))
        compact = _re.sub(r"\s+", "", text)
        for match in _re.finditer(r"סעיף\d+(?:\([^)]*\))?", compact):
            label = match.group(0)
            if label not in labels:
                labels.append(label)
        sections[int(doc["id"])] = labels[:400]
    return sections


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    profiles = []
    for doc in manifest:
        text = (TEXT_ROOT / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
        profile = build_document_profile(doc, text)
        profile["id"] = doc["id"]
        profiles.append(profile)

    retriever = DocumentRetriever(profiles, document_sections=build_document_sections())
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    for case in cases:
        required = set(case.get("required_document_ids", []))
        if not required:
            continue
        results = retriever.retrieve(case["question"], top_k=5)
        selected = [r["document_id"] for r in results]
        hit_at = None
        for rank, doc_id in enumerate(selected, 1):
            if doc_id in required:
                hit_at = rank
                break
        rows.append(
            {
                "id": case["id"],
                "required": sorted(required),
                "selected": selected,
                "hit_at": hit_at,
                "recall_at_3": 1 if hit_at and hit_at <= 3 else 0,
                "recall_at_5": 1 if hit_at and hit_at <= 5 else 0,
            }
        )

    metrics = {
        "cases": len(rows),
        "recall_at_3": round(sum(r["recall_at_3"] for r in rows) / max(len(rows), 1), 3),
        "recall_at_5": round(sum(r["recall_at_5"] for r in rows) / max(len(rows), 1), 3),
        "mean_rank_of_first_hit": round(
            sum(r["hit_at"] for r in rows if r["hit_at"]) / max(sum(1 for r in rows if r["hit_at"]), 1), 2
        ),
    }
    out = Path("results/challenger_metrics.json")
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
