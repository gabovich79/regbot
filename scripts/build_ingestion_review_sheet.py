#!/usr/bin/env python3
"""Build a read-only, source-first ingestion review sheet for the corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from services.document_ingestion_service import build_ingestion_receipt
from services.document_service import extract_docx, extract_pdf

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
TEXT_ROOT = Path("eval/corpus-texts")
DEFAULT_ARTIFACT_DIRS = [
    Path("artifacts/recovered_sources"),
    Path("/Users/guygabovich/Downloads"),
    Path("/tmp"),
]


def classify_source_availability(
    document: dict[str, Any], artifact_dirs: list[Path]
) -> dict[str, str]:
    """Classify provenance without pretending an extracted text is an original."""
    source_ref = str(document.get("source_ref") or "")
    candidates = [Path(source_ref).name, f"{document['id']}.pdf", f"{document['id']}.docx", f"{document['id']}.html"]
    if document.get("original_path"):
        candidates.append(Path(str(document["original_path"])).name)
    for directory in artifact_dirs:
        for name in candidates:
            candidate = directory / name
            if candidate.is_file():
                return {"status": "local_original", "path": str(candidate)}
    if source_ref.startswith(("https://", "http://")):
        return {"status": "fetchable_url", "url": source_ref}
    return {"status": "needs_reupload"}


def source_text_for_review(
    document: dict[str, Any], source: dict[str, str], fallback_path: Path
) -> tuple[str, str]:
    """Extract from recovered/original bytes first; label export-only fallback."""
    if source["status"] == "local_original":
        artifact = Path(source["path"])
        recovered_text = artifact.parent / f"{document['id']}.txt"
        if recovered_text.is_file():
            return recovered_text.read_text(encoding="utf-8", errors="replace"), "recovered_source_text"
        suffix = artifact.suffix.lower()
        if suffix == ".pdf":
            return extract_pdf(str(artifact)), "original_pdf_extraction"
        if suffix in {".docx", ".doc"}:
            return extract_docx(str(artifact)), "original_docx_extraction"
    if fallback_path.exists():
        return fallback_path.read_text(encoding="utf-8", errors="replace"), "legacy_text_export"
    return "", "missing_text"


def _receipt_status(document: dict[str, Any], text: str, source: dict[str, str]) -> dict[str, Any]:
    if source["status"] == "local_original":
        content = Path(source["path"]).read_bytes()
        original_path = source["path"]
        checksum = hashlib.sha256(content).hexdigest()
    elif source["status"] == "fetchable_url":
        original_path = None
        checksum = None
    else:
        original_path = None
        checksum = None
    return build_ingestion_receipt(
        document,
        text,
        original_path=original_path,
        source_checksum=checksum,
    )


def build_review(
    manifest: list[dict[str, Any]], text_root: Path, artifact_dirs: list[Path]
) -> dict[str, Any]:
    rows = []
    for document in manifest:
        text_path = text_root / f"{document['id']}.txt"
        source = classify_source_availability(document, artifact_dirs)
        text, text_origin = source_text_for_review(document, source, text_path)
        receipt = _receipt_status(document, text, source)

        if source["status"] == "needs_reupload":
            action_status = "needs_reupload"
        elif source["status"] == "fetchable_url":
            action_status = "needs_source_fetch"
        elif receipt["status"] == "validated":
            action_status = "ready_for_reingest"
        else:
            action_status = receipt["status"]

        rows.append(
            {
                "document_id": document["id"],
                "title": document["title"],
                "action_status": action_status,
                "source_availability": source,
                "text_origin": text_origin,
                "canonical_title": receipt["profile"]["canonical_title"],
                "official_number": receipt["profile"]["official_number"],
                "identity_evidence": receipt["profile"]["identity_evidence"],
                "integrity_status": receipt["integrity"]["status"],
                "integrity_reasons": receipt["integrity"]["reasons"],
                "validation_errors": receipt["validation_errors"],
                "counts": receipt["counts"],
            }
        )
    counts = Counter(row["action_status"] for row in rows)
    return {"documents": rows, "summary": dict(sorted(counts.items()))}


def markdown_report(review: dict[str, Any]) -> str:
    lines = [
        "# Ingestion Review Queue",
        "",
        "**Read-only local audit. No production DB, metadata, or files were changed.**",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in review["summary"].items())
    lines.extend(
        [
            "",
            "## Exceptions requiring action",
            "",
            "| ID | Current title | Status | Integrity / validation reason | Source action |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in review["documents"]:
        if row["action_status"] == "ready_for_reingest":
            continue
        reasons = ", ".join(row["integrity_reasons"] + row["validation_errors"]) or "—"
        source = row["source_availability"]
        source_action = source.get("url") or source.get("path") or "upload original"
        lines.append(
            f"| {row['document_id']} | {row['title']} | {row['action_status']} | {reasons} | {source_action} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a source-first corpus ingestion review")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--text-root", type=Path, default=TEXT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, action="append", default=DEFAULT_ARTIFACT_DIRS)
    parser.add_argument("--output", type=Path, default=Path("results/ingestion_review.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/INGESTION_REVIEW.md"))
    args = parser.parse_args()

    review = build_review(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.text_root,
        args.artifact_dir,
    )
    args.output.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown_report(review), encoding="utf-8")
    print(json.dumps(review["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
