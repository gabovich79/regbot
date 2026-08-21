"""Read-only health audit for a RegBot document corpus."""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path


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

    documents = []
    for row in rows:
        document = dict(row)
        text_path = Path(document["text_path"])
        text_exists = text_path.is_file()
        extraction_chars = len(text_path.read_text(encoding="utf-8")) if text_exists else 0
        issues = []
        if not text_exists:
            issues.append("missing_text_file")
        elif extraction_chars == 0:
            issues.append("empty_extraction")
        if document["is_active"] and document["chunk_count"] == 0:
            issues.append("no_chunks")

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
            "issues": issues,
        })

    issue_counts = Counter(issue for document in documents for issue in document["issues"])
    active_documents = [document for document in documents if document["is_active"]]
    return {
        "summary": {
            "documents": len(documents),
            "active_documents": len(active_documents),
            "indexed_documents": sum(document["chunk_count"] > 0 for document in active_documents),
            "unindexed_documents": sum(document["chunk_count"] == 0 for document in active_documents),
            "missing_text_files": issue_counts["missing_text_file"],
            "total_chunks": sum(document["chunk_count"] for document in documents),
        },
        "issues": dict(sorted(issue_counts.items())),
        "documents": documents,
    }
