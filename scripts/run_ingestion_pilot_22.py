#!/usr/bin/env python3
"""Run an isolated ingestion-contract pilot against the original document 22."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from models import database
from services.document_ingestion_service import (
    build_ingestion_receipt,
    persist_ingestion_receipt,
)
from services.document_service import extract_docx

SOURCE = Path("/Users/guygabovich/Downloads/regulation_h_2016-9-11.docx")
CACHE = Path("results/challenger_embeddings_cache_local.json")
OUTPUT = Path("results/ingestion_pilot_22.json")


async def run() -> dict:
    if not SOURCE.exists():
        raise FileNotFoundError(f"missing pilot source: {SOURCE}")

    content = SOURCE.read_bytes()
    text = extract_docx(str(SOURCE))
    checksum = hashlib.sha256(content).hexdigest()
    original_db_path = database.DB_PATH

    with tempfile.TemporaryDirectory(prefix="regbot-ingestion-pilot-") as temp_dir:
        database.DB_PATH = str(Path(temp_dir) / "pilot.db")
        try:
            await database.init_db()
            pilot_document_id = await database.add_document(
                "העברת כספים בין קופות גמל - תיקון",
                "docx",
                str(SOURCE),
                str(Path(temp_dir) / "22.txt"),
                len(text.split()),
            )
            document = {
                "id": pilot_document_id,
                "title": "העברת כספים בין קופות גמל - תיקון",
                "source_type": "docx",
                "source_ref": str(SOURCE),
                "document_type": "חוזר",
                "lifecycle_status": "current",
                "topic": "ניודים, העברות בין קופות",
            }
            receipt = build_ingestion_receipt(
                document,
                text,
                original_path=str(SOURCE),
                source_checksum=checksum,
            )
            if not CACHE.exists():
                raise FileNotFoundError(f"missing local challenger cache: {CACHE}")
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            node_embeddings = {
                hashlib.sha256(node["raw_text"].encode("utf-8")).hexdigest(): node["embedding"]
                for node in cached["nodes"]
                if int(node["document_id"]) == 22 and node.get("embedding")
            }
            db = await database.get_db()
            try:
                persisted = await persist_ingestion_receipt(
                    db,
                    receipt,
                    node_embeddings_by_hash=node_embeddings,
                )
                profile_fts = await (
                    await db.execute(
                        "SELECT COUNT(*) AS count FROM document_profiles_fts WHERE document_id = ?",
                        (pilot_document_id,),
                    )
                ).fetchone()
                node_fts = await (
                    await db.execute(
                        "SELECT COUNT(*) AS count FROM document_nodes_fts WHERE document_id = ?",
                        (pilot_document_id,),
                    )
                ).fetchone()
                embedded_nodes = await (
                    await db.execute(
                        "SELECT COUNT(*) AS count FROM document_nodes WHERE document_id = ? AND embedding IS NOT NULL",
                        (pilot_document_id,),
                    )
                ).fetchone()
                receipt_row = await (
                    await db.execute(
                        "SELECT status FROM document_ingestion_receipts WHERE document_id = ?",
                        (pilot_document_id,),
                    )
                ).fetchone()
            finally:
                await db.close()
        finally:
            database.DB_PATH = original_db_path

    report = {
        "corpus_document_id": 22,
        "pilot_document_id": pilot_document_id,
        "source": str(SOURCE),
        "status": receipt["status"],
        "checksum": checksum,
        "canonical_title": receipt["profile"]["canonical_title"],
        "official_number": receipt["profile"]["official_number"],
        "identity_evidence": receipt["profile"]["identity_evidence"],
        "integrity": receipt["integrity"],
        "keywords": receipt["keywords"],
        "counts": receipt["counts"],
        "persisted": persisted,
        "cached_embedding_hashes": len(node_embeddings),
        "profile_fts_records": profile_fts["count"],
        "node_fts_records": node_fts["count"],
        "embedded_node_records": embedded_nodes["count"],
        "stored_receipt_status": receipt_row["status"],
        "validation_errors": receipt["validation_errors"],
        "isolation": "temporary SQLite database only; production was not written",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    asyncio.run(run())
