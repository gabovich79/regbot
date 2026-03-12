"""
Simple keyword-based document retrieval for RAG.
Scores documents by relevance to the user's question using TF-IDF-like scoring,
then returns the most relevant documents that fit within the token budget.
"""

import re
import math
from collections import Counter


# Common Hebrew stop words to ignore during scoring
HEBREW_STOP_WORDS = {
    "של", "את", "על", "עם", "או", "לא", "כי", "אם", "גם", "מה",
    "זה", "זו", "אל", "הם", "הן", "אנו", "היא", "הוא", "כל", "יש",
    "אין", "בין", "לפי", "אך", "רק", "עד", "כן", "לו", "כך", "אחר",
    "היה", "היו", "להיות", "כאשר", "אשר", "בו", "מן", "ביותר", "ידי",
    "שם", "שלא", "למה", "איך", "מתי", "איפה", "כמה", "לפני", "אחרי",
    "תחת", "מעל", "בגלל", "בשביל", "למען", "לגבי", "בתוך", "מחוץ",
}

ENGLISH_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "about",
    "and", "or", "but", "if", "not", "no", "so", "it", "its", "this",
    "that", "these", "those", "he", "she", "we", "they", "i", "you",
}

ALL_STOP_WORDS = HEBREW_STOP_WORDS | ENGLISH_STOP_WORDS


def _tokenize(text: str) -> list[str]:
    """Split text into words, removing punctuation and stop words."""
    # Split on whitespace and punctuation
    words = re.findall(r'[\w\u0590-\u05FF]+', text.lower())
    return [w for w in words if w not in ALL_STOP_WORDS and len(w) > 1]


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for Hebrew/mixed text.
    Hebrew uses ~1 token per 2-3 characters in Claude's tokenizer.
    We use len/3 as a safe estimate to avoid exceeding API limits."""
    return max(len(text) // 3, len(text.split()) * 2)


def score_document(question_tokens: list[str], doc_text: str) -> float:
    """
    Score a document's relevance to the question.
    Uses TF (term frequency in doc) * IDF-like boost (rarer terms score higher).
    """
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens or not question_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    score = 0.0
    for token in question_tokens:
        tf = doc_counter.get(token, 0) / doc_len
        if tf > 0:
            # Simple boost: longer query terms are likely more specific/important
            specificity_boost = min(len(token) / 3.0, 2.0)
            score += tf * specificity_boost

    return score


def retrieve_relevant_documents(
    question: str,
    documents: list[dict],
    max_tokens: int,
    min_score: float = 0.0,
) -> tuple[list[dict], list[str]]:
    """
    Select the most relevant documents for the given question,
    fitting within max_tokens budget.

    Args:
        question: User's question
        documents: List of {"title": str, "text": str} dicts
        max_tokens: Maximum total tokens for selected documents
        min_score: Minimum relevance score to include (0 = include all if space permits)

    Returns:
        (selected_docs, skipped_titles) - selected documents and names of skipped ones
    """
    if not documents:
        return [], []

    question_tokens = _tokenize(question)

    # Score each document
    scored_docs = []
    for doc in documents:
        score = score_document(question_tokens, doc["text"])
        tokens = _estimate_tokens(doc["text"])
        scored_docs.append({
            "title": doc["title"],
            "text": doc["text"],
            "score": score,
            "tokens": tokens,
        })

    # Sort by relevance score (highest first)
    scored_docs.sort(key=lambda d: d["score"], reverse=True)

    # Select documents that fit within budget, prioritizing high-relevance docs
    selected = []
    skipped = []
    used_tokens = 0

    for doc in scored_docs:
        if used_tokens + doc["tokens"] <= max_tokens:
            selected.append({"title": doc["title"], "text": doc["text"]})
            used_tokens += doc["tokens"]
        else:
            # Try truncation for partially fitting high-relevance documents
            remaining = max_tokens - used_tokens
            if remaining > 3000 and doc["score"] > 0:
                # Truncate to fit — ~4 chars per token for Hebrew
                max_chars = remaining * 3  # ~3 chars per token for Hebrew
                selected.append({
                    "title": doc["title"] + " (קטוע)",
                    "text": doc["text"][:max_chars] + "\n[... המסמך נחתך עקב מגבלת גודל ...]",
                })
                used_tokens += remaining
            else:
                skipped.append(doc["title"])

    return selected, skipped
