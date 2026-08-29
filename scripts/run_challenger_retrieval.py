#!/usr/bin/env python3
"""Compare flat retrieval vs hierarchical challenger on the eval cases."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.document_retriever import DocumentRetriever
from services.document_profile_service import build_document_profile
from services.legal_parser import build_legal_tree
from services.section_retriever import SectionRetriever

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
TEXT_ROOT = Path("/tmp/regbot-hierarchical-corpus/texts")
FLAT_CASES = Path("eval/retrieval_cases.jsonl")
HIER_CASES = Path("eval/hierarchical_cases.jsonl")


def load_cases() -> list[dict]:
    cases = []
    for path in (FLAT_CASES, HIER_CASES):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases


def load_profiles() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    profiles = []
    for doc in manifest:
        text = (TEXT_ROOT / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
        profile = build_document_profile(doc, text)
        profile["id"] = doc["id"]
        profiles.append(profile)
    return profiles


def load_sections(profiles: list[dict]) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    nodes = []
    node_id = 1
    for doc in manifest:
        text = (TEXT_ROOT / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
        paragraphs = [line for line in text.splitlines() if line.strip()]
        tree = build_legal_tree(paragraphs, doc)
        stack = [(tree, None)]
        while stack:
            node, parent_id = stack.pop()
            nodes.append(
                {
                    "id": node_id,
                    "document_id": node["document_id"],
                    "parent_id": parent_id,
                    "node_type": node["node_type"],
                    "heading": node["heading"],
                    "raw_text": node["raw_text"],
                    "page_start": None,
                }
            )
            current_id = node_id
            node_id += 1
            for child in reversed(node["children"]):
                stack.append((child, current_id))
    return nodes


def main() -> None:
    profiles = load_profiles()
    nodes = load_sections(profiles)
    doc_retriever = DocumentRetriever(profiles)
    section_retriever = SectionRetriever(nodes)
    cases = load_cases()

    results = []
    for case in cases:
        question = case["question"]
        selected = doc_retriever.retrieve(question, top_k=5)
        selected_ids = [d["document_id"] for d in selected]
        sections = section_retriever.retrieve(question, document_ids=selected_ids, top_k=5)
        results.append(
            {
                "id": case["id"],
                "question": question,
                "selected_documents": selected_ids,
                "top_sections": [
                    {"document_id": s["document_id"], "heading": s.get("heading", "")}
                    for s in sections[:3]
                ],
                "expected_documents": case.get("required_document_ids", []),
            }
        )

    out = Path("results/challenger_retrieval_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
