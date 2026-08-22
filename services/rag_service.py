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
from config import OPENAI_API_KEY, EMBEDDING_MODEL, RAG_MAX_CONTEXT_TOKENS
from services.validity import document_validity_status

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


# A superseded/expired document loses a small fused-relevance amount (roughly
# two RRF ranks), so a current document surfaces first when relevance is close.
# Explicit references (a circular number in the question) still dominate via the
# lexical +100 term, keeping historical lookups intact.
VALIDITY_PENALTY = 0.001
VALIDITY_STATUS_LABELS = {
    "current": "",
    "superseded": "הוחלף במסמך עדכני",
    "expired": "פג תוקף",
}


def format_document_header(chunk: dict) -> str:
    """Render an evidence-block header, flagging superseded/expired documents."""
    parts = [f"=== {chunk['document_title']}"]
    ref = chunk.get("document_ref")
    if ref:
        parts.append(ref)
    effective = chunk.get("effective_date")
    if effective:
        parts.append(f"תוקף: {effective}")
    label = VALIDITY_STATUS_LABELS.get(chunk.get("validity_status") or "", "")
    if label:
        parts.append(f"[{label}]")
    return " | ".join(parts) + " ==="



def fit_context_blocks_to_budget(blocks: list[str], max_tokens: int) -> list[str]:
    """Keep ordered evidence blocks that fit an explicit per-question budget."""
    selected = []
    used_tokens = 0
    for block in blocks:
        tokens = len(EMBEDDING_ENCODING.encode(block))
        if used_tokens + tokens > max_tokens:
            continue
        selected.append(block)
        used_tokens += tokens
    return selected


def _hebrew_token_variants(token: str) -> set[str]:
    """Handle common Hebrew prefixes without pretending to be a full lemmatizer."""
    variants = {token}
    if len(token) > 3 and token[0] in "והבכלמש":
        variants.add(token[1:])
    return variants


def _lexical_score(question: str, chunk: dict) -> float:
    """Score exact regulatory identifiers and title terms above semantic drift."""
    normalized_question = question.lower()
    title = (chunk.get("document_title") or "").lower()
    source_ref = (chunk.get("document_ref") or "").lower()
    content = (chunk.get("content") or "").lower()
    circular_references = re.findall(r"\d{4}-\d+-\d+", normalized_question)
    references = re.findall(r"(?:סעיף|תקנה)\s+\d+(?:\([^\s)]+\))*", normalized_question)
    percentages = re.findall(r"\d+%", normalized_question)

    score = 0.0
    for reference in circular_references:
        if reference in title or reference in source_ref:
            score += 100.0
    for reference in references:
        if reference in title or reference in source_ref or reference in (chunk.get("section_header") or "").lower() or reference in content:
            score += 50.0
    for percentage in percentages:
        if percentage in content:
            score += 20.0

    query_tokens = [
        token for token in re.findall(r"[\w\u0590-\u05FF]+", normalized_question)
        if len(token) > 2 and not token.isdigit()
    ]
    for token in query_tokens:
        variants = _hebrew_token_variants(token)
        if any(variant in title for variant in variants):
            score += 5.0
        if any(variant in source_ref for variant in variants):
            score += 3.0
        if any(variant in content for variant in variants):
            score += 0.2
    return score



LEGAL_QUERY_GROUPS = (
    ("העברה", ("העברה", "העברת כספים")),
    ("שעבוד", ("שעבוד",)),
    ("עיקול", ("עיקול",)),
)


def build_retrieval_queries(question: str) -> list[str]:
    """Expand multi-issue legal questions into focused retrieval queries."""
    normalized = question.lower()
    queries = [question]

    if "זכויות עמית" in normalized:
        matched = [label for label, terms in LEGAL_QUERY_GROUPS if any(term in normalized for term in terms)]
        if len(matched) >= 2:
            queries.extend(
                f"{label} זכויות עמית חוק הפיקוח על קופות גמל סעיף 25" for label in matched
            )

    if "הלוואה" in normalized and "קרן השתלמות" in normalized:
        queries.extend([
            "הלוואה לעמית מקרן השתלמות תנאים חוזר 2016-9-17 סעיף 8(ד)",
            "הלוואה כנגד כספים נזילים ולא נזילים קרן השתלמות 50% 80% שבע שנים",
        ])

    return queries


def force_domain_evidence_chunks(
    question: str,
    ranked_chunks: list[dict],
    candidate_chunks: list[dict],
) -> list[dict]:
    """Force direct numeric loan-rule evidence into context for loan questions."""
    normalized = question.lower()
    if "הלוואה" not in normalized or "קרן השתלמות" not in normalized:
        return ranked_chunks
    markers = ("50%", "80%", "50 אחוז", "80 אחוז", "שבע שנים", "7 שנים", "8(ד)")
    exact = [
        chunk for chunk in candidate_chunks
        if any(marker in (chunk.get("content") or "").lower() for marker in markers)
        or any(marker in (chunk.get("section_header") or "").lower() for marker in markers)
    ]
    selected = list(ranked_chunks)
    selected_ids = {(chunk["document_id"], chunk.get("chunk_index")) for chunk in selected}
    for chunk in exact:
        key = (chunk["document_id"], chunk.get("chunk_index"))
        if key not in selected_ids:
            selected.insert(0, chunk)
            selected_ids.add(key)
    return selected


def infer_document_authority(chunk: dict) -> str:
    """Infer legal authority for legacy chunks that lack document_type metadata."""
    explicit = (chunk.get("document_type") or "").strip()
    if explicit:
        return explicit
    title = (chunk.get("document_title") or "").lower()
    if "חוק" in title:
        return "חוק"
    if "תקנ" in title:
        return "תקנה"
    if "חוזר" in title:
        return "חוזר"
    if "הכרעה" in title:
        return "הכרעה"
    return "אחר"


def authority_bonus(question: str, chunk: dict) -> float:
    """Prefer primary legal authority for rights/prohibition questions."""
    legal_terms = ("זכויות", "העברה", "שעבוד", "עיקול", "איסור", "מותר", "אסור")
    if not any(term in question for term in legal_terms):
        return 0.0
    return {
        "חוק": 0.040,
        "תקנה": 0.020,
        "חוזר": 0.010,
        "הכרעה": 0.008,
        "אחר": 0.0,
    }.get(infer_document_authority(chunk), 0.0)

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
        validity_status = chunk.get("validity_status")
        penalty = VALIDITY_PENALTY if validity_status in ("superseded", "expired") else 0.0
        score = (
            1 / (rrf_k + dense_rank[key])
            + 1 / (rrf_k + lexical_rank[key])
            + 0.01 * lexical_score[key]
            + authority_bonus(question, chunk)
            - penalty
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


async def _retrieve_top_chunks(question: str, db, top_k: int = 20) -> list[dict]:
    """Embed and rank original plus focused queries across active documents."""
    retrieval_queries = build_retrieval_queries(question)
    query_vectors = []
    for query in retrieval_queries:
        q_response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_vectors.append(np.array(q_response.data[0].embedding))

    # Load all chunks from active documents.
    cursor = await db.execute("""
        SELECT dc.*, d.superseded_by, d.valid_until,
               d.effective_date AS doc_effective_date,
               d.document_type
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.is_active = 1
    """)
    rows = await cursor.fetchall()

    if not rows:
        return []

    # Score every chunk against each focused query. A chunk that matches more
    # than one issue gets a small consensus bonus; the original question still
    # remains in the query set and anchors overall relevance.
    scored = []
    for row in rows:
        row_dict = dict(row)
        doc_effective = row_dict.pop("doc_effective_date", None)
        if doc_effective:
            row_dict["effective_date"] = doc_effective
        row_dict["validity_status"] = document_validity_status(row_dict)
        vec = np.array(json.loads(row_dict["embedding"]))
        similarities = [
            float(np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-10))
            for q_vec in query_vectors
        ]
        score = max(similarities) + 0.01 * max(0, sum(value >= 0.35 for value in similarities) - 1)
        scored.append((score, row_dict))

    ranked_chunks = rank_hybrid_chunks(question, scored, max_per_document=3)
    ranked_chunks = force_domain_evidence_chunks(
        question,
        ranked_chunks,
        [chunk for _, chunk in scored],
    )
    return ranked_chunks[:top_k]


async def retrieve_ranked_documents(question: str, db, top_k: int = 20) -> list[dict]:
    """Return documents ordered by relevance, for evaluation and metrics."""
    top_chunks = await _retrieve_top_chunks(question, db, top_k=top_k)
    ranked: list[dict] = []
    seen: set[int] = set()
    for chunk in top_chunks:
        doc_id = chunk["document_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append({
            "document_id": doc_id,
            "title": chunk["document_title"],
            "source_ref": chunk.get("document_ref"),
            "validity_status": chunk.get("validity_status"),
        })
    return ranked


async def retrieve_relevant_chunks(
    question: str,
    db,
    top_k: int = 20,
    context_window: int = 1,
    max_context_tokens: int = RAG_MAX_CONTEXT_TOKENS,
) -> str:
    """
    Retrieve the most relevant chunks for a question using embedding similarity.
    Returns a formatted string ready to send to Claude.
    """
    top_chunks = await _retrieve_top_chunks(question, db, top_k=top_k)

    if not top_chunks:
        return ""

    # Carry document-level validity and effective date into the expanded fetch,
    # which reads document_chunks alone (no documents join).
    validity_by_doc = {
        chunk["document_id"]: chunk.get("validity_status") for chunk in top_chunks
    }
    effective_by_doc = {
        chunk["document_id"]: chunk.get("effective_date") for chunk in top_chunks
    }

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
        expanded = dict(row)
        expanded["validity_status"] = validity_by_doc.get(expanded["document_id"])
        doc_effective = effective_by_doc.get(expanded["document_id"])
        if doc_effective:
            expanded["effective_date"] = doc_effective
        by_doc[expanded["document_id"]].append(expanded)

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
        parts.append(format_document_header(doc_chunks[0]))
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

    return "\n\n".join(fit_context_blocks_to_budget(parts, max_context_tokens))
