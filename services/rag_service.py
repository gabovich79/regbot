"""
Embedding-based RAG service for regulatory documents.
Chunks documents by section, creates embeddings via OpenAI,
and retrieves relevant chunks at query time using cosine similarity.
"""

import re
import json
import numpy as np
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SECTION_PATTERN = re.compile(
    r'(?:^|\n)(?:סעיף\s+\d+|פרק\s+[א-ת]|\d+\.\s|\d+\)\s)',
    re.MULTILINE
)


def chunk_regulatory_document(text: str, doc_metadata: dict) -> list[dict]:
    """Split a regulatory document into chunks by section headers."""
    splits = [(m.start(), m.group().strip()) for m in SECTION_PATTERN.finditer(text)]

    if len(splits) < 3:
        return _chunk_by_paragraph(text, doc_metadata)

    chunks = []
    for i, (start, header) in enumerate(splits):
        end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
        content = text[start:end].strip()
        if len(content) < 40:
            continue
        chunks.append({
            "content": content,
            "section_header": header,
            "chunk_index": i,
            "document_id": doc_metadata["id"],
            "document_title": doc_metadata["title"],
            "document_ref": doc_metadata.get("source_ref", ""),
            "effective_date": doc_metadata.get("effective_date", ""),
            "topic": doc_metadata.get("topic", ""),
        })
    return chunks


def _chunk_by_paragraph(text: str, doc_metadata: dict) -> list[dict]:
    """Fallback — split by paragraphs of ~800 words with 100-word overlap."""
    words = text.split()
    size, overlap, chunks = 800, 100, []
    i = 0
    while i < len(words):
        content = " ".join(words[i:i + size])
        chunks.append({
            "content": content,
            "section_header": f"קטע {len(chunks) + 1}",
            "chunk_index": len(chunks),
            "document_id": doc_metadata["id"],
            "document_title": doc_metadata["title"],
            "document_ref": doc_metadata.get("source_ref", ""),
            "effective_date": doc_metadata.get("effective_date", ""),
            "topic": doc_metadata.get("topic", ""),
        })
        i += size - overlap
    return chunks


async def embed_and_store_chunks(chunks: list[dict], db) -> int:
    """
    Create embeddings for chunks and store in DB.
    Deletes old chunks for the same document_id first.
    Returns number of chunks saved.
    """
    if not chunks:
        return 0

    doc_id = chunks[0]["document_id"]

    # Delete previous chunks for this document (re-index case)
    await db.execute(
        "DELETE FROM document_chunks WHERE document_id = ?", (doc_id,)
    )

    BATCH_SIZE = 100
    total_saved = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )

        for chunk, emb_obj in zip(batch, response.data):
            await db.execute("""
                INSERT INTO document_chunks
                (document_id, content, section_header, chunk_index,
                 document_title, document_ref, effective_date, topic, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["document_id"],
                chunk["content"],
                chunk["section_header"],
                chunk["chunk_index"],
                chunk["document_title"],
                chunk["document_ref"],
                chunk["effective_date"],
                chunk["topic"],
                json.dumps(emb_obj.embedding)
            ))
            total_saved += 1

    await db.commit()
    return total_saved


async def retrieve_relevant_chunks(
    question: str,
    db,
    top_k: int = 20,
    context_window: int = 1
) -> str:
    """
    Retrieve the most relevant chunks for a question using embedding similarity.
    Returns a formatted string ready to send to Claude.
    """
    # Embed the question
    q_response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    q_vec = np.array(q_response.data[0].embedding)

    # Load all chunks from active documents
    cursor = await db.execute("""
        SELECT dc.* FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.is_active = 1
    """)
    rows = await cursor.fetchall()

    if not rows:
        return ""

    # Cosine similarity scoring
    scored = []
    for row in rows:
        row_dict = dict(row)
        vec = np.array(json.loads(row_dict["embedding"]))
        score = float(
            np.dot(q_vec, vec) /
            (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-10)
        )
        scored.append((score, row_dict))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:top_k]]

    if not top_chunks:
        return ""

    # Context expansion — add neighboring chunks
    ids_to_fetch = set()
    for chunk in top_chunks:
        doc_id = chunk["document_id"]
        idx = chunk["chunk_index"]
        for offset in range(-context_window, context_window + 1):
            ids_to_fetch.add((doc_id, idx + offset))

    # Fetch expanded chunks
    placeholders = ",".join(
        f"'{d}_{i}'" for d, i in ids_to_fetch
    )
    cursor = await db.execute(f"""
        SELECT * FROM document_chunks
        WHERE (document_id || '_' || chunk_index) IN ({placeholders})
        ORDER BY document_id, chunk_index
    """)
    expanded_rows = await cursor.fetchall()

    # Group by document
    from collections import defaultdict
    by_doc = defaultdict(list)
    for row in expanded_rows:
        by_doc[row["document_id"]].append(dict(row))

    # Maintain order by relevance (most relevant document first)
    seen_docs = []
    for chunk in top_chunks:
        if chunk["document_id"] not in seen_docs:
            seen_docs.append(chunk["document_id"])

    # Build formatted context string
    parts = []
    for doc_id in seen_docs:
        doc_chunks = sorted(by_doc.get(doc_id, []), key=lambda x: x["chunk_index"])
        if not doc_chunks:
            continue
        first = doc_chunks[0]
        header = (
            f"=== {first['document_title']}"
            f"{' | ' + first['document_ref'] if first['document_ref'] else ''}"
            f"{' | תוקף: ' + first['effective_date'] if first['effective_date'] else ''}"
            f" ==="
        )
        parts.append(header)
        for c in doc_chunks:
            section = f"[{c['section_header']}]\n" if c["section_header"] else ""
            parts.append(f"{section}{c['content']}")

    return "\n\n".join(parts)
