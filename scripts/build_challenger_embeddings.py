"""Build embeddings cache for hierarchical challenger (profiles + nodes)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, DOCUMENTS_DIR
from services.document_profile_service import build_document_profile
from services.legal_parser import build_legal_tree
from services.embeddings import embed_texts

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
CACHE_PATH = Path("results/challenger_embeddings_cache.json")
EMBEDDING_ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_NODE_EMBEDDING_TOKENS = 6_400


def split_node_text_for_embedding(text: str) -> list[str]:
    """Split an oversized node at provider-safe token boundaries losslessly."""
    tokens = EMBEDDING_ENCODING.encode(text)
    if len(tokens) <= MAX_NODE_EMBEDDING_TOKENS:
        return [text]
    return [
        EMBEDDING_ENCODING.decode(tokens[start:start + MAX_NODE_EMBEDDING_TOKENS])
        for start in range(0, len(tokens), MAX_NODE_EMBEDDING_TOKENS)
    ]


def text_root() -> Path:
    """Use the deployed documents dir when DATA_DIR is /var/data (Render)."""
    if os.environ.get("DATA_DIR"):
        return Path(DOCUMENTS_DIR)
    return Path("/tmp/regbot-hierarchical-corpus/texts")


def load_documents() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    root = text_root()
    docs = []
    for doc in manifest:
        text = (root / f"{doc['id']}.txt").read_text(encoding="utf-8", errors="replace")
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
        # Stack carries the nearest persisted ancestor. A short heading may be
        # skipped as an embedding node, but its children must still be visited.
        stack = [(tree, None)]
        while stack:
            node, persisted_parent_id = stack.pop()
            raw_text = node.get("raw_text", "")
            persisted_node_id = persisted_parent_id
            if len(raw_text.strip()) >= 60:
                parts = split_node_text_for_embedding(raw_text)
                first_persisted_id = None
                for part_index, part_text in enumerate(parts, 1):
                    persisted_node_id = node_id
                    if first_persisted_id is None:
                        first_persisted_id = persisted_node_id
                    heading = node.get("heading", "")
                    if len(parts) > 1:
                        heading = f"{heading} — חלק {part_index}/{len(parts)}"
                    nodes.append(
                        {
                            "id": persisted_node_id,
                            "document_id": int(node.get("document_id", doc["id"])),
                            "parent_id": persisted_parent_id,
                            "node_type": node.get("node_type", "section"),
                            "heading": heading,
                            "raw_text": part_text,
                            "page_start": None,
                        }
                    )
                    node_id += 1
                persisted_node_id = first_persisted_id
            for child in reversed(node.get("children", [])):
                stack.append((child, persisted_node_id))
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
