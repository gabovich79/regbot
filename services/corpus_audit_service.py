"""Read-only health audit for a RegBot document corpus."""

from __future__ import annotations

import sqlite3
import hashlib
from collections import Counter
from pathlib import Path

from services.validity import document_validity_status


def audit_corpus(db_path: str | Path) -> dict:
    """Return document, extraction, and indexing health from a RegBot SQLite DB."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            """
            SELECT d.*, COUNT(dc.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks dc ON dc.document_id = d.id
            GROUP BY d.id
            ORDER BY d.id
            """
        ).fetchall()
    finally:
        db.close()

    from services.document_profile_service import build_document_profile
    from services.document_integrity_service import assess_document_integrity
    documents = []
    checksums = {}
    for row in rows:
        document = dict(row)
        text_path = Path(document["text_path"])
        text_exists = text_path.is_file()
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_exists else ''
        extraction_chars = len(text)
        issues = []
        if not text_exists:
            issues.append("missing_text_file")
        elif extraction_chars == 0:
            issues.append("empty_extraction")
        if document["is_active"] and document["chunk_count"] == 0:
            issues.append("no_chunks")
        original = document.get('original_path')
        original_exists = bool(original and Path(original).is_file())
        if not original_exists:
            issues.append('missing_original')
        profile = build_document_profile(document,text)
        integrity = assess_document_integrity(document,text,profile)
        issues.extend(reason for reason in integrity['reasons'] if reason not in issues)
        text_checksum = hashlib.sha256(text.encode('utf-8')).hexdigest() if text else None
        duplicate_of = checksums.get(text_checksum) if text_checksum else None
        if duplicate_of is not None:
            issues.append('duplicate_extracted_text')
        elif text_checksum:
            checksums[text_checksum] = document['id']
        if not document.get('effective_date'):
            issues.append('unknown_effective_date')
        if profile.get('draft_markers') or profile.get('document_type') == 'טיוטה':
            issues.append('draft_status_requires_review')

        documents.append({
            "id": document["id"],
            "title": document["title"],
            "source_type": document["source_type"],
            "source_ref": document["source_ref"],
            "is_active": bool(document["is_active"]),
            "token_count": document["token_count"] or 0,
            "chunk_count": document["chunk_count"],
            "text_path": str(text_path),
            "text_file_exists": text_exists,
            "extraction_chars": extraction_chars,
            "effective_date": document.get("effective_date"),
            "valid_until": document.get("valid_until"),
            "superseded_by": document.get("superseded_by"),
            "topic": document.get("topic"),
            "document_type": document.get("document_type"),
            "lifecycle_status": document.get("lifecycle_status") or "current",
            "validity_status": document_validity_status(document),
            "issues": issues,
            "text_checksum": text_checksum,
            "duplicate_of": duplicate_of,
            "original_exists": original_exists,
            "identity_evidence": profile['identity_evidence'],
            "review_status": 'requires_source_review' if issues else 'machine_checked_pending_human',
        })

    issue_counts = Counter(issue for document in documents for issue in document["issues"])
    active_documents = [document for document in documents if document["is_active"]]
    validity = Counter(document["validity_status"] for document in active_documents)
    undated = sum(
        1 for document in active_documents if not document["effective_date"]
    )
    return {
        "summary": {
            "documents": len(documents),
            "active_documents": len(active_documents),
            "indexed_documents": sum(document["chunk_count"] > 0 for document in active_documents),
            "unindexed_documents": sum(document["chunk_count"] == 0 for document in active_documents),
            "missing_text_files": issue_counts["missing_text_file"],
            "total_chunks": sum(document["chunk_count"] for document in documents),
            "validity": {
                "current": validity["current"],
                "superseded": validity["superseded"],
                "expired": validity["expired"],
                "undated": undated,
            },
        },
        "issues": dict(sorted(issue_counts.items())),
        "documents": documents,
    }
