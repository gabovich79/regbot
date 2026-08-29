#!/usr/bin/env python3
"""Measure extraction quality of the current pipeline on representative docs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from docx import Document
import fitz


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def measure_pdf(path: Path) -> dict:
    start = time.monotonic()
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        pages.append({"page": page.number + 1, "text": clean_text(text), "chars": len(text)})
    elapsed = time.monotonic() - start
    full = clean_text("\n".join(p["text"] for p in pages))
    hebrew_chars = len(re.findall(r"[\u0590-\u05FF]", full))
    return {
        "path": str(path),
        "kind": "pdf",
        "pages": len(pages),
        "chars": len(full),
        "hebrew_chars": hebrew_chars,
        "nonempty_pages": sum(1 for p in pages if p["chars"] > 20),
        "elapsed_s": round(elapsed, 3),
    }


def measure_docx(path: Path) -> dict:
    start = time.monotonic()
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    elapsed = time.monotonic() - start
    full = clean_text("\n".join(paragraphs))
    hebrew_chars = len(re.findall(r"[\u0590-\u05FF]", full))
    return {
        "path": str(path),
        "kind": "docx",
        "paragraphs": len(paragraphs),
        "chars": len(full),
        "hebrew_chars": hebrew_chars,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    corpus = [
        Path("/Users/guygabovich/Downloads/regulation_h_2016-9-11.docx"),
        Path("/tmp/age-3.pdf"),
        Path("/Users/guygabovich/Downloads/LegalInformation_kesher_פקודת מס הכנסה [נוסח חדש] - לא מרובד.pdf"),
    ]
    results = []
    for path in corpus:
        if not path.exists():
            continue
        results.append(measure_pdf(path) if path.suffix.lower() == ".pdf" else measure_docx(path))

    out = Path("results/extractor_measurements.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
