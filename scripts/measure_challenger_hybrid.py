#!/usr/bin/env python3
"""Hybrid (dense+lexical) hierarchical retrieval evaluation.

Reads embeddings cache produced by build_challenger_embeddings.py and measures
document Recall@3/@5 on the hierarchical cases.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import EMBEDDING_MODEL
from services.document_retriever import DocumentRetriever

CACHE = Path("results/challenger_embeddings_cache.json")
CASES = Path("eval/hierarchical_cases.jsonl")


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def main() -> int:
    if not CACHE.exists():
        print(json.dumps({"error": "cache missing; run build_challenger_embeddings.py first"}))
        return 1
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    profiles = cache["profiles"]
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    lexical = DocumentRetriever(profiles, document_sections={})
    rows = []
    for case in cases:
        required = set(case.get("required_document_ids", []))
        if not required:
            continue
        question = case["question"]
        # Lexical order first (already includes official numbers/sections).
        lex = lexical.retrieve(question, top_k=10)
        lex_rank = {d["document_id"]: i for i, d in enumerate(lex)}

        dense_scores = []
        for p in profiles:
            dense_scores.append((cosine(p["embedding"], _embed_query(question, profiles)), p))
        dense_scores.sort(key=lambda item: item[0], reverse=True)
        dense_rank = {p["document_id"]: i for i, (_, p) in enumerate(dense_scores)}

        fused = []
        for p in profiles:
            did = int(p["document_id"])
            rrf = 1 / (60 + lex_rank.get(did, 100)) + 1 / (60 + dense_rank.get(did, 100))
            fused.append((rrf, did))
        fused.sort(key=lambda item: item[0], reverse=True)
        selected = [did for _, did in fused[:5]]

        hit_at = next((i + 1 for i, did in enumerate(selected) if did in required), None)
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
    }
    out = Path("results/challenger_hybrid_metrics.json")
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _embed_query(question: str, profiles: list[dict]) -> list[float]:
    """Embed the question itself; fall back to a profile-mean proxy offline."""
    try:
        from services.embeddings import embed_texts

        vectors = asyncio.run(embed_texts([question]))
        return vectors[0]
    except Exception:
        from services.document_retriever import _terms

        terms = _terms(question)
        hits = [
            p
            for p in profiles
            if terms & _terms(
                " ".join(
                    str(p.get(field) or "")
                    for field in ("canonical_title", "topics", "profile_summary")
                )
            )
        ]
        if not hits:
            return [0.0] * len(profiles[0]["embedding"])
        dim = len(profiles[0]["embedding"])
        vec = np.zeros(dim, dtype=float)
        for p in hits[:5]:
            vec += np.asarray(p["embedding"], dtype=float)
        vec /= len(hits[:5])
        return vec.tolist()


if __name__ == "__main__":
    raise SystemExit(main())
