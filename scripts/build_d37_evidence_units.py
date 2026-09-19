#!/usr/bin/env python3
"""Build the D37 evidence-unit artifact from an original PDF and annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.document_service import extract_pdf_pages
from services.evidence_unit_service import build_evidence_units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.pdf.read_bytes()
    config = json.loads(args.annotations.read_text(encoding="utf-8"))
    pages = extract_pdf_pages(str(args.pdf))
    units = build_evidence_units(
        pages,
        source_id=config["source_id"],
        source_checksum=hashlib.sha256(source_bytes).hexdigest(),
        annotations=config["annotations"],
    )
    notes = {item["evidence_id"]: item.get("verification_note") for item in config["annotations"]}
    for unit in units:
        unit["verification_note"] = notes[unit["evidence_id"]]

    artifact = {
        "schema_version": "d37-evidence-units/v1",
        "source": {
            "source_id": config["source_id"],
            "document_title": config["document_title"],
            "local_file_name": args.pdf.name,
            "source_checksum": hashlib.sha256(source_bytes).hexdigest(),
            "page_count": len(pages),
        },
        "evidence_units": units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evidence_units": len(units), "source_checksum": artifact["source"]["source_checksum"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
