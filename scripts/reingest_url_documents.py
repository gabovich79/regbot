"""Re-ingest URL-backed documents while retaining PDF page provenance.

Dry run (no writes):
    python scripts/reingest_url_documents.py

Apply to every URL-backed document:
    python scripts/reingest_url_documents.py --apply

Apply to one reviewed document:
    python scripts/reingest_url_documents.py --apply --document-id 23
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import _index_document
from models.database import (
    get_all_documents,
    init_db,
    update_document_extraction,
    update_document_source_artifact,
)
from services.document_service import (
    fetch_url_document,
    normalize_source_url,
    save_original_document,
    save_document_text,
)
from services.token_utils import estimate_tokens


async def reingest_url_documents(*, apply: bool, document_ids: set[int] | None):
    await init_db()
    documents = await get_all_documents(active_only=True)
    documents = [doc for doc in documents if doc["source_type"] == "url"]
    if document_ids:
        documents = [doc for doc in documents if doc["id"] in document_ids]

    print(f"Found {len(documents)} URL-backed active documents")
    if not apply:
        print("Dry run only — use --apply to download, save and re-index documents.")

    succeeded = failed = skipped = 0
    for document in documents:
        source_url = normalize_source_url(document.get("source_ref", ""))
        if not source_url:
            skipped += 1
            print(f"  [{document['id']}] SKIP: no usable URL in source_ref")
            continue

        if not apply:
            print(f"  [{document['id']}] WOULD RE-INGEST: {source_url}")
            continue

        try:
            text, pages, source_bytes, extension = await fetch_url_document(source_url)
            if not text.strip():
                raise ValueError("the source returned no extractable text")
            original_path, source_checksum = save_original_document(
                document["id"], extension, source_bytes
            )
            await update_document_source_artifact(
                document["id"], original_path=original_path, checksum=source_checksum
            )
            text_path = save_document_text(document["id"], text)
            await update_document_extraction(
                document["id"], text_path=text_path, token_count=estimate_tokens(text)
            )
            chunks = await _index_document(
                document["id"], document["title"], source_url, text, pages=pages
            )
            page_note = f", {len(pages)} PDF pages" if pages is not None else ""
            succeeded += 1
            print(f"  [{document['id']}] OK: {chunks} chunks{page_note}")
        except Exception as error:
            failed += 1
            print(f"  [{document['id']}] ERROR: {error}")

    print(f"\nDone. succeeded: {succeeded}; failed: {failed}; skipped: {skipped}")
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Re-ingest URL-backed RegBot documents")
    parser.add_argument("--apply", action="store_true", help="perform writes and re-indexing")
    parser.add_argument(
        "--document-id", action="append", type=int,
        help="limit to one or more specific document IDs",
    )
    args = parser.parse_args()
    asyncio.run(
        reingest_url_documents(
            apply=args.apply,
            document_ids=set(args.document_id) if args.document_id else None,
        )
    )


if __name__ == "__main__":
    main()
