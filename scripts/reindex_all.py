"""
Re-index all active documents: chunk + embed.
Run once after deploy or when changing chunking strategy:
    python scripts/reindex_all.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.database import get_db, init_db
from services.rag_service import chunk_regulatory_document, embed_and_store_chunks


async def reindex_all():
    await init_db()
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, title, source_ref, text_path FROM documents WHERE is_active = 1"
        )
        docs = await cursor.fetchall()
        print(f"Found {len(docs)} active documents to index")

        total_chunks = 0
        for doc in docs:
            doc = dict(doc)
            try:
                with open(doc["text_path"], "r", encoding="utf-8") as f:
                    text = f.read()
                metadata = {
                    "id": doc["id"],
                    "title": doc["title"],
                    "source_ref": doc.get("source_ref", ""),
                    "effective_date": None,
                    "topic": None,
                }
                chunks = chunk_regulatory_document(text, metadata)
                num = await embed_and_store_chunks(chunks, db)
                total_chunks += num
                print(f"  [{doc['id']}] {doc['title']} -> {num} chunks")
            except Exception as e:
                print(f"  [{doc['id']}] ERROR: {e}")

        print(f"\nDone. Total chunks indexed: {total_chunks}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(reindex_all())
