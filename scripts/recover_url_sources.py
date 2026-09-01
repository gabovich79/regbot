#!/usr/bin/env python3
"""Recover URL-backed source artifacts locally; dry-run by default."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from services.document_service import fetch_url_document

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
OUTPUT_DIR = Path("artifacts/recovered_sources")
REPORT = Path("results/url_source_recovery.json")
DEFAULT_ARTIFACT_DIRS = [Path("/Users/guygabovich/Downloads"), Path("/tmp")]


def has_local_artifact(document: dict) -> bool:
    source_ref = str(document.get("source_ref") or "")
    candidates = [Path(source_ref).name]
    if document.get("original_path"):
        candidates.append(Path(str(document["original_path"])).name)
    return any((directory / name).is_file() for directory in DEFAULT_ARTIFACT_DIRS for name in candidates)


async def recover(*, apply: bool) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    url_documents = [
        document
        for document in manifest
        if str(document.get("source_ref") or "").startswith(("https://", "http://"))
        and not has_local_artifact(document)
    ]
    rows = []
    for document in url_documents:
        row = {
            "document_id": document["id"],
            "title": document["title"],
            "url": document["source_ref"],
            "status": "planned",
        }
        if apply:
            try:
                text, pages, content, source_type = await fetch_url_document(document["source_ref"])
                suffix = "docx" if source_type == "docx" else source_type
                artifact_path = OUTPUT_DIR / f"{document['id']}.{suffix}"
                text_path = OUTPUT_DIR / f"{document['id']}.txt"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_bytes(content)
                text_path.write_text(text, encoding="utf-8")
                row.update(
                    {
                        "status": "recovered",
                        "artifact_path": str(artifact_path),
                        "text_path": str(text_path),
                        "checksum": hashlib.sha256(content).hexdigest(),
                        "text_characters": len(text),
                        "page_count": len(pages) if pages is not None else None,
                        "source_type": source_type,
                    }
                )
            except Exception as error:
                row.update({"status": "recovery_failed", "error": str(error)})
        rows.append(row)
    summary = {}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return {"apply": apply, "summary": summary, "documents": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover official URL-backed source artifacts")
    parser.add_argument("--apply", action="store_true", help="download and save local artifacts")
    args = parser.parse_args()
    report = asyncio.run(recover(apply=args.apply))
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
