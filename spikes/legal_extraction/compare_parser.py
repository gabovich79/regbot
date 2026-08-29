#!/usr/bin/env python3
"""Compare old flat chunking vs new legal hierarchy parser on real docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.legal_parser import HEADING_PATTERN, build_legal_tree


def section_pattern_matches(text: str) -> int:
    return len(HEADING_PATTERN.findall(text))


def paragraph_sources(path: Path) -> list[str]:
    if path.suffix.lower() == ".docx":
        doc = Document(str(path))
        return [p.text for p in doc.paragraphs if p.text.strip()]
    pdf = fitz.open(str(path))
    paragraphs = []
    for page in pdf:
        for line in page.get_text().splitlines():
            line = line.strip()
            if line:
                paragraphs.append(line)
    return paragraphs


def main() -> None:
    corpus = {
        "circular-docx": Path("/Users/guygabovich/Downloads/regulation_h_2016-9-11.docx"),
        "age-track-pdf": Path("/tmp/age-3.pdf"),
    }
    report = {}
    for name, path in corpus.items():
        paragraphs = paragraph_sources(path)
        text = "\n".join(paragraphs)
        old_sections = section_pattern_matches(text)
        tree = build_legal_tree(paragraphs, {"id": 1, "title": name})
        headings = [
            node["heading"]
            for node in tree["children"]
            if node["node_type"] in {"chapter", "section", "subsection"}
        ]
        report[name] = {
            "paragraphs": len(paragraphs),
            "old_section_matches": old_sections,
            "new_headings": len(headings),
            "sample_headings": headings[:12],
        }
    out = Path("results/parser_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
