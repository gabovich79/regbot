"""Build a source-derived, validate-before-active ingestion receipt.

This module is deliberately pure: it performs no database writes, embeddings,
or production activation. Persistence must consume only a `validated` receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from services.document_integrity_service import assess_document_integrity
from services.document_profile_service import build_document_profile, save_document_profile
from services.legal_parser import build_legal_tree
from services.token_utils import estimate_tokens


KEYWORD_STOPWORDS = {
    "של",
    "על",
    "את",
    "עם",
    "או",
    "לפי",
    "הוא",
    "היא",
    "אלה",
    "הוראות",
    "כללי",
    "גמל",
    "קופות",
    "קופת",
}


def _keywords(profile: dict[str, Any]) -> list[str]:
    """Create deterministic routing keywords from source-derived identity fields."""
    source = " ".join(
        [
            str(profile.get("canonical_title") or ""),
            str(profile.get("official_number") or ""),
            " ".join(profile.get("topics") or []),
            " ".join(profile.get("heading_outline") or []),
        ]
    )
    keywords = {
        token
        for token in re.findall(r"[\w\u0590-\u05FF-]+", source.lower())
        if len(token) > 2 and token not in KEYWORD_STOPWORDS
    }
    return sorted(keywords)[:80]


def _flatten_meaningful_nodes(tree: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    stack: list[tuple[dict[str, Any], str, str | None]] = [(tree, "document", None)]
    ordinal = 0
    while stack:
        node, path, parent_path = stack.pop()
        raw_text = str(node.get("raw_text") or "").strip()
        heading = str(node.get("heading") or "").strip()
        if node.get("node_type") != "document" and len(raw_text) >= 60:
            ordinal += 1
            node_path = f"{path}/{ordinal}"
            retrieval_text = " | ".join(
                part
                for part in (
                    profile.get("canonical_title"),
                    profile.get("official_number"),
                    heading,
                    raw_text,
                )
                if part
            )
            nodes.append(
                {
                    "node_path": node_path,
                    "parent_path": parent_path,
                    "node_type": node.get("node_type", "section"),
                    "section_label": heading or None,
                    "heading": heading or None,
                    "raw_text": raw_text,
                    "retrieval_text": retrieval_text,
                    "ordinal": ordinal,
                    "text_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                }
            )
            next_parent_path = node_path
        else:
            next_parent_path = parent_path
        for child_index, child in reversed(list(enumerate(node.get("children", []), start=1))):
            stack.append((child, f"{path}.{child_index}", next_parent_path))
    return nodes


def build_ingestion_receipt(
    document: dict[str, Any],
    text: str,
    *,
    original_path: str | None,
    source_checksum: str | None,
    pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Produce all deterministic ingestion artifacts and a validation decision."""
    text = text or ""
    profile = build_document_profile(document, text)
    profile["keywords"] = _keywords(profile)
    integrity = assess_document_integrity(document, text, profile)
    paragraphs = [line for line in text.splitlines() if line.strip()]
    tree = build_legal_tree(paragraphs, document, page_map=pages)
    nodes = _flatten_meaningful_nodes(tree, profile)

    validation_errors: list[str] = []
    if not original_path:
        validation_errors.append("original_source_missing")
    if not source_checksum:
        validation_errors.append("source_checksum_missing")
    if not text.strip():
        validation_errors.append("empty_extracted_text")
    if integrity["status"] == "failed":
        validation_errors.extend(integrity["reasons"])
    if text.strip() and not profile.get("identity_evidence"):
        validation_errors.append("identity_evidence_missing")
    if text.strip() and not nodes:
        validation_errors.append("meaningful_legal_nodes_missing")

    validation_errors = list(dict.fromkeys(validation_errors))
    if validation_errors:
        status = "needs_reupload" if any(
            error in {
                "original_source_missing",
                "source_checksum_missing",
                "empty_extracted_text",
                "extraction_binary",
                "extraction_hebrew_reversed",
                "meaningful_legal_nodes_missing",
            }
            for error in validation_errors
        ) else "needs_human_review"
    elif integrity["status"] == "warning":
        status = "needs_human_review"
    else:
        status = "validated"

    return {
        "document_id": int(document["id"]),
        "status": status,
        "source": {
            "original_path": original_path,
            "checksum": source_checksum,
            "source_type": document.get("source_type"),
            "source_ref": document.get("source_ref"),
            "page_count": len(pages) if pages is not None else None,
        },
        "profile": profile,
        "integrity": integrity,
        "keywords": profile["keywords"],
        "nodes": nodes,
        "counts": {
            "text_characters": len(text),
            "estimated_tokens": estimate_tokens(text),
            "meaningful_nodes": len(nodes),
            "fts_profile_records": 1 if profile.get("canonical_title") else 0,
            "fts_node_records": len(nodes),
        },
        "validation_errors": validation_errors,
        "pipeline_version": 1,
    }


async def persist_ingestion_receipt(
    db,
    receipt: dict[str, Any],
    *,
    node_embeddings_by_hash: dict[str, list[float]] | None = None,
) -> dict[str, int]:
    """Atomically persist profile, node tree, FTS projections, and receipt."""
    document_id = int(receipt["document_id"])
    await db.execute("SAVEPOINT ingestion_receipt_persist")
    try:
        await save_document_profile(
            db,
            receipt["profile"],
            receipt["integrity"],
            commit=False,
        )
        await db.execute("DELETE FROM document_nodes_fts WHERE document_id = ?", (document_id,))
        await db.execute("DELETE FROM document_nodes WHERE document_id = ?", (document_id,))

        node_ids: dict[str, int] = {}
        node_records = 0
        embedded_node_records = 0
        for node in receipt["nodes"]:
            parent_id = node_ids.get(node.get("parent_path") or "")
            vector = (node_embeddings_by_hash or {}).get(node["text_hash"])
            embedding_json = json.dumps(vector) if vector is not None else None
            cursor = await db.execute(
                """
                INSERT INTO document_nodes (
                    document_id, parent_id, node_type, node_path, section_label,
                    heading, raw_text, retrieval_text, page_start, page_end,
                    ordinal, text_hash, embedding, is_evidence, index_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, 1, 1)
                """,
                (
                    document_id,
                    parent_id,
                    node["node_type"],
                    node["node_path"],
                    node.get("section_label"),
                    node.get("heading"),
                    node["raw_text"],
                    node["retrieval_text"],
                    node["ordinal"],
                    node["text_hash"],
                    embedding_json,
                ),
            )
            node_id = int(cursor.lastrowid)
            node_ids[node["node_path"]] = node_id
            await db.execute(
                """
                INSERT INTO document_nodes_fts
                    (node_id, document_id, heading, section_label, retrieval_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    document_id,
                    node.get("heading") or "",
                    node.get("section_label") or "",
                    node["retrieval_text"],
                ),
            )
            node_records += 1
            if embedding_json is not None:
                embedded_node_records += 1

        await db.execute(
            """
            INSERT INTO document_ingestion_receipts (
                document_id, status, receipt_json, validation_errors_json,
                pipeline_version, updated_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(document_id) DO UPDATE SET
                status=excluded.status,
                receipt_json=excluded.receipt_json,
                validation_errors_json=excluded.validation_errors_json,
                pipeline_version=excluded.pipeline_version,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                document_id,
                receipt["status"],
                json.dumps(receipt, ensure_ascii=False),
                json.dumps(receipt["validation_errors"], ensure_ascii=False),
                receipt["pipeline_version"],
            ),
        )
        await db.execute("RELEASE SAVEPOINT ingestion_receipt_persist")
        await db.commit()
        return {
            "profile_records": 1,
            "node_records": node_records,
            "fts_node_records": node_records,
            "embedded_node_records": embedded_node_records,
        }
    except BaseException:
        await db.execute("ROLLBACK TO SAVEPOINT ingestion_receipt_persist")
        await db.execute("RELEASE SAVEPOINT ingestion_receipt_persist")
        raise
