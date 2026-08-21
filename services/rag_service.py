"""
Embedding-based RAG service for regulatory documents.
Chunks documents by section, creates embeddings via OpenAI,
and retrieves relevant chunks at query time using cosine similarity.
"""

import re
import json
import numpy as np
import tiktoken
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SECTION_PATTERN = re.compile(
    r'(?:^|\n)(?:סעיף\s+\d+|פרק\s+[א-ת]|\d+\.\s|\d+\)\s)',
    re.MULTILINE
)

# `text-embedding-3-large` rejects inputs at 8,192 tokens. Reserve room below
# that hard limit so headings and overlap never cause an ingestion failure.
EMBEDDING_ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_EMBEDDING_TOKENS = 7_000
CHUNK_OVERLAP_WORDS = 80


def format_chunk_citation(chunk: dict) -> str:
    """Create a stable, human-auditable citation ID for an evidence chunk."""
    document_id = chunk["document_id"]
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start is not None and page_end is not None:
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        return f"D{document_id}-P{page_label}"
    return f"D{document_id}-C{chunk.get('chunk_index', 0) + 1}"


def _lexical_score(question: str, chunk: dict) -> float:
    """Score exact regulatory identifiers and title terms above semantic drift."""
    normalized_question = question.lower()
    title = (chunk.get("document_title") or "").lower()
    source_ref = (chunk.get("document_ref") or "").lower()
    content = (chunk.get("content") or "").lower()
    references = re.findall(r"\d{4}-\d+-\d+", normalized_question)

    score = 0.0
    for reference in references:
        if reference in title or reference in source_ref:
            score += 100.0

    query_tokens = [
        token for token in re.findall(r"[\w\u0590-\u05FF]+", normalized_question)
        if len(token) > 2 and not token.isdigit()
    ]
    for token in query_tokens:
        if token in title:
            score += 5.0
        if token in source_ref:
            score += 3.0
        if token in content:
            score += 0.2
    return score


def rank_hybrid_chunks(
    question: str,
    dense_scored_chunks: list[tuple[float, dict]],
    *,
    max_per_document: int = 3,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse dense and lexical rankings, then retain evidence diversity."""
    if not dense_scored_chunks:
        return []

    dense_ranked = sorted(dense_scored_chunks, key=lambda item: item[0], reverse=True)
    lexical_ranked = sorted(
        dense_scored_chunks,
        key=lambda item: _lexical_score(question, item[1]),
        reverse=True,
    )
    dense_rank = {id(chunk): rank for rank, (_, chunk) in enumerate(dense_ranked, start=1)}
    lexical_rank = {id(chunk): rank for rank, (_, chunk) in enumerate(lexical_ranked, start=1)}
    lexical_score = {id(chunk): _lexical_score(question, chunk) for _, chunk in dense_scored_chunks}

    fused = []
    for _, chunk in dense_scored_chunks:
        key = id(chunk)
        score = (
            1 / (rrf_k + dense_rank[key])
            + 1 / (rrf_k + lexical_rank[key])
            + 0.01 * lexical_score[key]
        )
        fused.append((score, chunk))
    fused.sort(key=lambda item: item[0], reverse=True)

    selected = []
    document_counts = {}
    for _, chunk in fused:
        document_id = chunk["document_id"]
        if document_counts.get(document_id, 0) >= max_per_document:
            continue
        selected.append(chunk)
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
    return selected


def _split_to_embedding_limit(text: str) -> list[str]:
    """Split text into overlapping chunks accepted by the embedding API."""
    if len(EMBEDDING_ENCODING.encode(text)) <= MAX_EMBEDDING_TOKENS:
        return [text]

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        low, high, best_end = start + 1, len(words), start
        while low <= high:
            middle = (low + high) // 2
            candidate = " ".join(words[start:middle])
            if len(EMBEDDING_ENCODING.encode(candidate)) <= MAX_EMBEDDING_TOKENS:
                best_end = middle
                low = middle + 1
            else:
                high = middle - 1

        if best_end == start:
            # A pathological single token/word must still respect the provider
            # limit. Losing a little lexical structure beats losing the whole
            # document to a 400 response.
            token_ids = EMBEDDING_ENCODING.encode(" ".join(words[start:]))
            chunks.extend(
                EMBEDDING_ENCODING.decode(token_ids[i:i + MAX_EMBEDDING_TOKENS])
                for i in range(0, len(token_ids), MAX_EMBEDDING_TOKENS)
            )
            break

        chunks.append(" ".join(words[start:best_end]))
        if best_end == len(words):
            break
        start = max(best_end - CHUNK_OVERLAP_WORDS, start + 1)

    return chunks


def _new_chunk(content: str, section_header: str, chunk_index: int, doc_metadata: dict) -> dict:
    return {
        "content": content,
        "section_header": section_header,
        "chunk_index": chunk_index,
        "document_id": doc_metadata["id"],
        "document_title": doc_metadata["title"],
        "document_ref": doc_metadata.get("source_ref", ""),
        "effective_date": doc_metadata.get("effective_date", ""),
        "topic": doc_metadata.get("topic", ""),
        "page_start": doc_metadata.get("page_start"),
        "page_end": doc_metadata.get("page_end"),
    }


def chunk_regulatory_document(text: str, doc_metadata: dict) -> list[dict]:
    """Split by legal section, then enforce the embedding provider's token cap."""
    splits = [(match.start(), match.group().strip()) for match in SECTION_PATTERN.finditer(text)]
    if len(splits) < 3:
        return _chunk_by_paragraph(text, doc_metadata)

    chunks = []
    for index, (start, header) in enumerate(splits):
        end = splits[index + 1][0] if index + 1 < len(splits) else len(text)
        content = text[start:end].strip()
        if len(content) < 40:
            continue
        section_parts = _split_to_embedding_limit(content)
        for part_index, part in enumerate(section_parts, start=1):
            label = header if len(section_parts) == 1 else f"{header} — חלק {part_index}"
            chunks.append(_new_chunk(part, label, len(chunks), doc_metadata))
    return chunks


def chunk_regulatory_pages(pages: list[dict], doc_metadata: dict) -> list[dict]:
    """Chunk page-wise extracted text while retaining citation page provenance."""
    chunks = []
    for page in pages:
        page_number = page["page_number"]
        page_metadata = {
            **doc_metadata,
            "page_start": page_number,
            "page_end": page_number,
        }
        for chunk in chunk_regulatory_document(page["text"], page_metadata):
            chunk["chunk_index"] = len(chunks)
            chunks.append(chunk)
    return chunks


def _chunk_by_paragraph(text: str, doc_metadata: dict) -> list[dict]:
    """Fallback for unstructured text, still constrained by embedding tokens."""
    return [
        _new_chunk(content, f"קטע {index}", index - 1, doc_metadata)
        for index, content in enumerate(_split_to_embedding_limit(text), start=1)
        if content.strip()
    ]


async def embed_and_store_chunks(chunks: list[dict], db) -> int:
    """
    Create all replacement embeddings before atomically replacing a document's
    existing chunks. If embedding or storage fails, the previous index remains
    available.
    """
    if not chunks:
        return 0

    doc_id = chunks[0]["document_id"]

    # Generate every replacement embedding before touching the existing index.
    # A provider outage must never turn a previously searchable document into an
    # empty document.
    # Cap batches below the provider's aggregate token ceiling. A document can
    # legitimately yield many near-7k chunks after safety splitting.
    BATCH_SIZE = 20
    embedded_chunks: list[tuple[dict, list[float]]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[chunk["content"] for chunk in batch],
        )
        if len(response.data) != len(batch):
            raise ValueError(
                f"expected {len(batch)} embeddings, got {len(response.data)}"
            )
        embedded_chunks.extend(
            (chunk, embedding.embedding)
            for chunk, embedding in zip(batch, response.data)
        )

    # Replace chunks in one transaction only after all provider calls succeeded.
    try:
        await db.execute("BEGIN")
        await db.execute(
            "DELETE FROM document_chunks WHERE document_id = ?", (doc_id,)
        )
        for chunk, embedding in embedded_chunks:
            await db.execute("""
                INSERT INTO document_chunks
                (document_id, content, section_header, chunk_index,
                 document_title, document_ref, effective_date, topic,
                 page_start, page_end, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["document_id"],
                chunk["content"],
                chunk["section_header"],
                chunk["chunk_index"],
                chunk["document_title"],
                chunk["document_ref"],
                chunk["effective_date"],
                chunk["topic"],
                chunk.get("page_start"),
                chunk.get("page_end"),
                json.dumps(embedding),
            ))
        await db.commit()
    except BaseException:
        # Cancellation can arrive after DELETE; always reset this connection's
        # transaction before propagating the interruption.
        await db.rollback()
        raise

    return len(embedded_chunks)


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

    top_chunks = rank_hybrid_chunks(question, scored, max_per_document=3)[:top_k]

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
            citation_id = format_chunk_citation(c)
            page_start = c.get("page_start")
            page_end = c.get("page_end")
            if page_start is not None and page_end is not None:
                page_label = f"עמוד {page_start}" if page_start == page_end else f"עמודים {page_start}-{page_end}"
            else:
                page_label = "עמוד לא זמין במאגר הישן"
            section_label = c["section_header"] or "קטע ללא כותרת"
            parts.append(
                f"[[SOURCE {citation_id} | מסמך: {c['document_title']} | "
                f"{page_label} | סעיף: {section_label} | URL: {c['document_ref']}]]\n"
                f"{c['content']}\n[[/SOURCE {citation_id}]]"
            )

    return "\n\n".join(parts)
