"""Build embeddings cache for hierarchical challenger (profiles + nodes).

Resumable: the cache is saved after every batch, keyed by text hash, so a
disconnected shell can simply re-run the same command and continue where the
previous run stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
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
CACHE_PATH = Path(
    os.environ.get(
        "CHALLENGER_CACHE_PATH",
        str(Path("results/challenger_embeddings_cache.json")),
    )
)
EMBEDDING_ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_NODE_EMBEDDING_TOKENS = 6_400
BATCH_SIZE = 16
BATCH_SLEEP_SECONDS = 1.0
# Stop cleanly after this many seconds so a disconnected shell never kills the
# process mid-write; re-running resumes from the atomic cache.
BUILD_TIME_BUDGET_SECONDS = float(os.environ.get("BUILD_TIME_BUDGET_SECONDS", "0"))


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
    """Resolve document text files: repo bundle > deploy disk > local cache."""
    repo_bundle = Path("eval/corpus-texts")
    if repo_bundle.exists() and any(repo_bundle.glob("*.txt")):
        return repo_bundle
    if os.environ.get("DATA_DIR"):
        return Path(DOCUMENTS_DIR)
    return Path("/tmp/regbot-hierarchical-corpus/texts")


def load_documents() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decisions = load_corpus_decisions()
    root = text_root()
    docs = []
    for doc in manifest:
        if not is_doc_active(doc, decisions):
            continue
        text = resolved_document_text(doc, root)
        docs.append({"doc": doc, "text": text})
    return docs


def load_corpus_decisions() -> dict:
    path = Path("eval/corpus_decisions.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def is_doc_active(doc: dict, decisions: dict) -> bool:
    """Apply the approved corpus decisions to cache construction."""
    doc_id = str(doc["id"])
    if doc_id in decisions.get("excluded_from_corpus", {}):
        return False
    if doc_id in decisions.get("duplicates", {}):
        return False
    unresolved = decisions.get("unresolved_source", {})
    if doc_id in unresolved:
        return False
    pending = decisions.get("metadata_review_pending", {})
    if doc_id in pending:
        return False
    return True


def resolved_document_text(doc: dict, fallback_root: Path) -> str:
    """Prefer the recovered original extraction, then the repo text export."""
    recovered = Path("artifacts/recovered_sources")
    candidates = [
        recovered / f"{doc['id']}.pdf",
        recovered / f"{doc['id']}.docx",
        recovered / f"{doc['id']}.doc",
    ]
    for artifact in candidates:
        if artifact.is_file():
            from services.document_service import extract_docx, extract_pdf

            suffix = artifact.suffix.lower()
            if suffix == ".pdf":
                return extract_pdf(str(artifact))
            if suffix == ".docx":
                return extract_docx(str(artifact))
            if suffix == ".doc":
                converted = artifact.with_suffix(".docx")
                if converted.is_file():
                    return extract_docx(str(converted))
    text_path = fallback_root / f"{doc['id']}.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="replace")
    return ""


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


def profile_text(profile: dict) -> str:
    return " | ".join(
        str(profile.get(field) or "")
        for field in ("canonical_title", "official_number", "topics", "profile_summary")
    )


def node_text(node: dict) -> str:
    return " | ".join(str(node.get(field) or "") for field in ("heading", "raw_text"))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing() -> dict:
    if not CACHE_PATH.exists():
        return {"profiles": [], "nodes": [], "profile_hashes": {}, "node_hashes": {}}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        profile_hashes = {
            _hash(profile_text(profile)): profile["embedding"]
            for profile in data.get("profiles", [])
            if profile.get("embedding")
        }
        node_hashes = {
            _hash(node_text(node)): node["embedding"]
            for node in data.get("nodes", [])
            if node.get("embedding")
        }
        return {
            "profiles": data.get("profiles", []),
            "nodes": data.get("nodes", []),
            "profile_hashes": profile_hashes,
            "node_hashes": node_hashes,
        }
    except Exception:
        return {"profiles": [], "nodes": [], "profile_hashes": {}, "node_hashes": {}}


def save_progress(
    profiles: list[dict],
    nodes: list[dict],
    profile_embeddings: dict[str, list[float]],
    node_embeddings: dict[str, list[float]],
) -> None:
    data = {
        "profiles": [
            {**profile, "embedding": profile_embeddings[_hash(profile_text(profile))]}
            for profile in profiles
            if _hash(profile_text(profile)) in profile_embeddings
        ],
        "nodes": [
            {**node, "embedding": node_embeddings[_hash(node_text(node))]}
            for node in nodes
            if _hash(node_text(node)) in node_embeddings
        ],
    }
    # Atomic write: never leave a partially-written cache behind if the
    # process is killed mid-write.
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(CACHE_PATH)


async def embed_missing(
    texts: list[str],
    existing: dict[str, list[float]],
    progress_label: str,
) -> dict[str, list[float]]:
    """Embed only texts missing from the existing hash map, saving every batch."""
    import time

    results = dict(existing)
    missing = [
        (text, _hash(text)) for text in texts if _hash(text) not in existing
    ]
    if not missing:
        return results
    started_at = time.monotonic()
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        vectors = await embed_texts([text for text, _ in batch])
        for (text, text_hash), vector in zip(batch, vectors):
            results[text_hash] = vector
        print(
            json.dumps(
                {
                    "progress": progress_label,
                    "done": len(results),
                    "pending": len(texts) - len(results),
                }
            ),
            flush=True,
        )
        if BUILD_TIME_BUDGET_SECONDS and (
            time.monotonic() - started_at >= BUILD_TIME_BUDGET_SECONDS
        ):
            print(
                json.dumps({"stopped": progress_label, "reason": "time_budget"}),
                flush=True,
            )
            break
        time.sleep(BATCH_SLEEP_SECONDS)
    return results


def main() -> int:
    docs = load_documents()
    profiles = build_profiles(docs)
    nodes = build_nodes(docs)

    existing = load_existing()
    profile_embeddings = dict(existing["profile_hashes"])
    node_embeddings = dict(existing["node_hashes"])

    async def run() -> None:
        nonlocal profile_embeddings, node_embeddings
        profile_embeddings = await embed_missing(
            [profile_text(p) for p in profiles],
            profile_embeddings,
            "profiles",
        )
        node_embeddings = await embed_missing(
            [node_text(n) for n in nodes],
            node_embeddings,
            "nodes",
        )
        save_progress(profiles, nodes, profile_embeddings, node_embeddings)

    asyncio.run(run())

    print(
        json.dumps(
            {
                "profiles": len(profiles),
                "nodes": len(nodes),
                "cache": str(CACHE_PATH),
                "embedded_nodes": len(node_embeddings),
            }
        )
    )
    return 0


def await_or_run(coro):
    return asyncio.run(coro)


if __name__ == "__main__":
    raise SystemExit(main())
