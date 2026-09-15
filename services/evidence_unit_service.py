"""Build verifiable evidence spans from page-scoped source annotations."""

from __future__ import annotations

import hashlib
from typing import Any


def build_evidence_units(
    pages: list[dict[str, Any]],
    *,
    source_id: str,
    source_checksum: str,
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract declared contiguous spans, refusing annotations that do not match.

    An annotation is document metadata, not a retrieval rule: it declares a
    canonical citation and text anchors that must be present in the declared
    source-page range.  The returned raw span is always copied directly from
    the page extraction and is content-addressed with ``span_hash``.
    """
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id is required")
    if len(source_checksum) != 64:
        raise ValueError("source_checksum must be a SHA-256 hex digest")

    by_page: dict[int, str] = {}
    for page in pages:
        number = page.get("page_number")
        text = page.get("text")
        if not isinstance(number, int) or not isinstance(text, str):
            raise ValueError("each page requires integer page_number and text")
        if number in by_page:
            raise ValueError(f"duplicate page_number: {number}")
        by_page[number] = text

    units: list[dict[str, Any]] = []
    for annotation in annotations:
        evidence_id = annotation["evidence_id"]
        section = annotation["section"]
        page_start = annotation["page_start"]
        page_end = annotation["page_end"]
        start_anchor = annotation["start_anchor"]
        verification_status = annotation["verification_status"]
        if not isinstance(page_start, int) or not isinstance(page_end, int) or page_end < page_start:
            raise ValueError(f"{evidence_id}: invalid declared page range")
        if not isinstance(start_anchor, str) or not start_anchor:
            raise ValueError(f"{evidence_id}: start_anchor is required")
        if not isinstance(verification_status, str) or not verification_status:
            raise ValueError(f"{evidence_id}: verification_status is required")

        missing_pages = [number for number in range(page_start, page_end + 1) if number not in by_page]
        if missing_pages:
            raise ValueError(f"{evidence_id}: missing declared source pages: {missing_pages}")

        declared_text = "\n".join(by_page[number] for number in range(page_start, page_end + 1))
        start = declared_text.find(start_anchor)
        if start < 0:
            raise ValueError(f"{evidence_id}: start_anchor is absent from declared source pages")

        end_anchor = annotation.get("end_exclusive_anchor")
        if end_anchor is None:
            end = len(declared_text)
        elif not isinstance(end_anchor, str) or not end_anchor:
            raise ValueError(f"{evidence_id}: end_exclusive_anchor must be a non-empty string")
        else:
            end = declared_text.find(end_anchor, start + len(start_anchor))
            if end < 0:
                raise ValueError(f"{evidence_id}: end_exclusive_anchor is absent after start_anchor")

        raw_text = declared_text[start:end].rstrip()
        if not raw_text:
            raise ValueError(f"{evidence_id}: annotation resolves to an empty source span")
        units.append(
            {
                "evidence_id": evidence_id,
                "source_id": source_id,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "source_checksum": source_checksum,
                "verification_status": verification_status,
                "raw_text": raw_text,
                "span_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            }
        )

    return units
