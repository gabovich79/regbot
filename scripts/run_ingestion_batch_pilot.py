#!/usr/bin/env python3
"""Validate all locally available originals through the ingestion contract.

Runs only against a temporary SQLite database. It never changes the production
DB, source metadata, or uploaded corpus.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from models import database
from scripts.build_ingestion_review_sheet import source_text_for_review
from services.document_ingestion_service import (
    build_ingestion_receipt,
    persist_ingestion_receipt,
)

MANIFEST = Path("eval/production_corpus_manifest_2026-08-29.json")
REVIEW = Path("results/ingestion_review.json")
CACHE = Path("results/challenger_embeddings_cache_local.json")
TEXT_ROOT = Path("eval/corpus-texts")
OUTPUT = Path("results/ingestion_batch_pilot.json")


def classify_index_readiness(receipt: dict, persisted: dict[str, int]) -> str:
    if receipt["status"] != "validated":
        return receipt["status"]
    if persisted["node_records"] != persisted["embedded_node_records"]:
        return "ready_for_embedding"
    return "ready_for_activation"


async def run() -> dict:
    manifest = {doc["id"]: doc for doc in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    embeddings = {}
    for node in cache["nodes"]:
        if node.get("embedding"):
            embeddings.setdefault(int(node["document_id"]), {})[
                hashlib.sha256(node["raw_text"].encode("utf-8")).hexdigest()
            ] = node["embedding"]

    original_db_path = database.DB_PATH
    rows = []
    with tempfile.TemporaryDirectory(prefix="regbot-ingestion-batch-") as temp_dir:
        database.DB_PATH = str(Path(temp_dir) / "batch.db")
        try:
            await database.init_db()
            db = await database.get_db()
            try:
                for review_row in review["documents"]:
                    if review_row["action_status"] not in {
                        "ready_for_reingest",
                        "needs_human_review",
                    }:
                        continue
                    document = manifest[review_row["document_id"]]
                    source = review_row["source_availability"]
                    source_path = Path(source["path"])
                    await db.execute(
                        """
                        INSERT INTO documents
                            (id, title, source_type, source_ref, text_path, token_count, index_status)
                        VALUES (?, ?, ?, ?, ?, ?, 'indexing')
                        """,
                        (
                            document["id"],
                            document["title"],
                            document["source_type"],
                            document["source_ref"],
                            str(TEXT_ROOT / f"{document['id']}.txt"),
                            0,
                        ),
                    )
                    text, text_origin = source_text_for_review(
                        document,
                        source,
                        TEXT_ROOT / f"{document['id']}.txt",
                    )
                    receipt = build_ingestion_receipt(
                        document,
                        text,
                        original_path=str(source_path),
                        source_checksum=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    )
                    persisted = await persist_ingestion_receipt(
                        db,
                        receipt,
                        node_embeddings_by_hash=embeddings.get(document["id"], {}),
                    )
                    readiness = classify_index_readiness(receipt, persisted)
                    rows.append(
                        {
                            "document_id": document["id"],
                            "title": document["title"],
                            "status": readiness,
                            "text_origin": text_origin,
                            "integrity": receipt["integrity"],
                            "validation_errors": receipt["validation_errors"],
                            "counts": receipt["counts"],
                            "persisted": persisted,
                        }
                    )
                await db.commit()
            finally:
                await db.close()
        finally:
            database.DB_PATH = original_db_path

    summary = Counter(row["status"] for row in rows)
    report = {"summary": dict(sorted(summary.items())), "documents": rows}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    asyncio.run(run())
