#!/usr/bin/env python3
"""Build source-derived document profiles; dry-run unless --write is explicit."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from models.database import get_db, init_db
from services.document_integrity_service import assess_document_integrity
from services.document_profile_service import build_document_profile, save_document_profile


DEFAULT_MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
DEFAULT_TEXT_ROOT = Path("/tmp/regbot-hierarchical-corpus/texts")
DEFAULT_OUTPUT = Path("results/document_profile_audit.json")


def build_profile_report(
    manifest: list[dict[str, Any]], text_root: Path, document_id: int | None = None
) -> dict[str, Any]:
    """Build a deterministic, read-only audit report from exported source text."""
    documents = []
    for document in manifest:
        if document_id is not None and int(document["id"]) != document_id:
            continue
        text_path = text_root / f"{document['id']}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        profile = build_document_profile(document, text)
        integrity = assess_document_integrity(document, text, profile)
        documents.append(
            {
                "document_id": int(document["id"]),
                "stored_title": document.get("title"),
                "text_path": str(text_path),
                "text_chars": len(text),
                "profile": profile,
                "integrity": integrity,
            }
        )
    counts = Counter(item["integrity"]["status"] for item in documents)
    return {
        "count": len(documents),
        "integrity_counts": {
            status: counts.get(status, 0)
            for status in ("verified", "warning", "failed", "pending")
        },
        "documents": documents,
    }


async def write_profiles(report: dict[str, Any]) -> None:
    """Persist profiles only when the operator explicitly passed --write."""
    await init_db()
    db = await get_db()
    try:
        for item in report["documents"]:
            await save_document_profile(db, item["profile"], item["integrity"])
    finally:
        await db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--document-id", type=int)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist profiles to the configured DB. Without this flag the run is read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = build_profile_report(manifest, args.text_root, args.document_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write:
        asyncio.run(write_profiles(report))
    print(
        json.dumps(
            {
                "mode": "write" if args.write else "dry-run",
                "count": report["count"],
                "integrity_counts": report["integrity_counts"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 1 if report["integrity_counts"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
