"""Re-index every active document with the currently configured embedding model.

Run after verifying that the embedding provider has usable quota:
    python scripts/reindex_all.py

The indexing service preserves existing chunks until replacement embeddings are
available, and records each document as ready or failed.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import _index_document
from models.database import get_all_documents, init_db
from services.document_service import load_document_text


async def reindex_all():
    await init_db()
    docs = await get_all_documents(active_only=True)
    print(f"Found {len(docs)} active documents to index")

    total_chunks = 0
    failed = 0
    for document in docs:
        try:
            text = load_document_text(document["text_path"])
            num_chunks = await _index_document(
                document["id"], document["title"], document.get("source_ref", ""), text
            )
            total_chunks += num_chunks
            print(f"  [{document['id']}] {document['title']} -> {num_chunks} chunks")
        except Exception as error:
            failed += 1
            print(f"  [{document['id']}] ERROR: {error}")

    print(f"\nDone. Total chunks indexed: {total_chunks}; failed documents: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(reindex_all())
