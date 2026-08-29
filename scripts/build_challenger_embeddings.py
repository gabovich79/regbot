"""Build embeddings cache for hierarchical challenger (profiles + nodes)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.document_profile_service import build_document_profile
from services.legal_parser import build_legal_tree
from services.embeddings import embed_texts

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
TEXT_ROOT = Path("/tmp/regbot-hierarchical-corpus/texts")
CACHE_PATH = Path("results/challenger_embeddings_cache.json")


def load_documents() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = []
    for doc in manifest:
        text = (TEXT_ROOT / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
        docs.append({"doc": doc, "text": text})
    return docs


def build_profiles(docs: list[dict]) -> list[dict]:
    profiles = []
    for item in docs:
        profile = build_document_profile(item["doc"], item["text"])
        profile["id"] = item["doc"]["id"]
        profiles.append(profile)
    return profiles


def build_nodes(docs: list[dict]) -> list[dict]:
    nodes = []
    node_id = 1
    for item in docs:
        doc = item["doc"]
        paragraphs = [line for line in item["text"].splitlines() if line.strip()]
        tree = build_legal_tree(paragraphs, doc)
        stack = [(tree, None)]
        while stack:
            node, parent_id = stack.pop()
            nodes.append(
                {
                    "id": node_id,
                    "document_id": int(node.get("document_id", doc["id"])),
                    "parent_id": parent_id,
                    "node_type": node.get("node_type", "section"),
                    "heading": node.get("heading", ""),
                    "raw_text": node.get("raw_text", ""),
                    "page_start": None,
                }
            )
            current_id = node_id
            node_id += 1
            for child in reversed(node.get("children", [])):
                stack.append((child, current_id))
    return nodes


def main() -> int:
    docs = load_documents()
    profiles = build_profiles(docs)
    nodes = build_nodes(docs)

    profile_texts = [
        " | ".join(
            str(profile.get(field) or "")
            for field in ("canonical_title", "official_number", "topics", "profile_summary")
        )
        for profile in profiles
    ]
    node_texts = [
        " | ".join(str(node.get(field) or "") for field in ("heading", "raw_text"))
        for node in nodes
    ]

    async def run() -> tuple[list[list[float]], list[list[float]]]:
        profile_vectors = await embed_texts(profile_texts)
        node_vectors = await embed_texts(node_texts)
        return profile_vectors, node_vectors

    profile_vectors, node_vectors = asyncio.run(run())

    CACHE_PATH.write_text(
        json.dumps(
            {
                "profiles": [
                    {**profile, "embedding": profile_vectors[i]} for i, profile in enumerate(profiles)
                ],
                "nodes": [
                    {**node, "embedding": node_vectors[i]} for i, node in enumerate(nodes)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "profiles": len(profiles),
                "nodes": len(nodes),
                "cache": str(CACHE_PATH),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
