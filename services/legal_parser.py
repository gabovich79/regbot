"""Deterministic Hebrew legal hierarchy parser.

The parser consumes paragraphs (DOCX paragraphs or PDF text lines) and emits a
node tree: document -> chapter -> section -> subsection -> paragraph/table.
It never invents raw text and never moves source bytes.
"""

from __future__ import annotations

import re
from typing import Any


HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:פרק\s+[א-ת]+[\w\s:-]*)"
    r"|(?:סעיף\s+\d+(?:\([^)]*\))?(?:[\s:.-].*)?)"
    r"|(?:\d+\s*\([א-ת]\)\s*\S)"
    r"|(?:\d+\s*[.)]\s*\S)"
    r"|(?:[א-ת]\)\s*\S)"
    r"|(?:תקנה\s+\d+(?:\([^)]*\))?)"
    r")",
    re.UNICODE,
)

SHORT_HEADINGS = {
    "כללי",
    "הגדרות",
    "ביטול חוזרים",
    "תחולה",
    "תחילה",
    "מסירת נתונים",
    "בוטל",
}


def _is_heading(text: str, *, allow_short_topic_heading: bool = True) -> bool:
    if not text:
        return False
    if text.endswith((".", ":", ";", ",")):
        return False
    if any(ch in text for ch in ("ð", "×", "|", "_")):
        return False
    if re.search(r"[\d]{3,}", text) and " " not in text:
        return False
    if text.startswith(("www.", "http", ":'", "רח'")):
        return False
    if text in SHORT_HEADINGS:
        return True
    if HEADING_PATTERN.match(text):
        return True
    # Short unnumbered topic lines are headings when they read as a title
    # (e.g. "טיפול בבקשת העברה", "הוראות מיוחדות לעניין העברה לקרן חדשה").
    if allow_short_topic_heading and 8 <= len(text) <= 60 and not any(marker in text for marker in ("- ", " – ")):
        return True
    return False


def _heading_level(text: str) -> int:
    if re.match(r"^פרק\s+[א-ת]+", text):
        return 1
    if re.match(r"^(?:סעיף|תקנה)\s+\d+", text):
        return 2
    if re.match(r"^\d+\s*\([א-ת]\)", text):
        return 3
    if re.match(r"^\d+\s*[.)]\s*\S", text):
        return 2
    if re.match(r"^[א-ת]\)", text):
        return 3
    # Unnumbered headings (כללי, הגדרות, topic titles) are siblings at
    # document level, exactly like chapters in a numbered law.
    return 1


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _paragraph_with_page(raw: str | dict[str, Any]) -> tuple[str, int | None, int | None]:
    """Normalize legacy strings and page-aware extraction records."""
    if isinstance(raw, str):
        return raw, None, None
    text = str(raw.get("text") or "")
    page_start = raw.get("page_start", raw.get("page_number"))
    page_end = raw.get("page_end", page_start)
    return text, int(page_start) if page_start is not None else None, int(page_end) if page_end is not None else None


def _include_page(node: dict[str, Any], page_start: int | None, page_end: int | None) -> None:
    """Expand a node's inclusive source-page range without inventing pages."""
    if page_start is None or page_end is None:
        return
    node["page_start"] = min(int(node.get("page_start", page_start)), page_start)
    node["page_end"] = max(int(node.get("page_end", page_end)), page_end)


def build_legal_tree(
    paragraphs: list[str | dict[str, Any]],
    document: dict[str, Any],
    *,
    page_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic hierarchy from a list of paragraph strings."""
    root: dict[str, Any] = {
        "node_type": "document",
        "document_id": int(document["id"]),
        "heading": document.get("title", ""),
        "raw_text": "",
        "children": [],
    }
    stack: list[dict[str, Any]] = [root]
    for raw in paragraphs:
        raw_text, page_start, page_end = _paragraph_with_page(raw)
        text = _clean(raw_text)
        if not text:
            continue
        is_page_extracted = isinstance(raw, dict) and any(
            key in raw for key in ("page_number", "page_start", "page_end")
        )
        if _is_heading(text, allow_short_topic_heading=not is_page_extracted):
            level = _heading_level(text)
            while len(stack) - 1 >= level:
                stack.pop()
            node: dict[str, Any] = {
                "node_type": "chapter" if level == 1 else "section" if level == 2 else "subsection",
                "document_id": int(document["id"]),
                "heading": text,
                "raw_text": text,
                "children": [],
            }
            stack[-1]["children"].append(node)
            stack.append(node)
        else:
            stack[-1]["raw_text"] = f"{stack[-1]['raw_text']}\n{text}".strip()
        for node in stack:
            _include_page(node, page_start, page_end)
    return root
