"""Document-level retriever over canonical document profiles (pure Python)."""

from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "מה",
    "האם",
    "איך",
    "איזה",
    "אילו",
    "לגבי",
    "של",
    "על",
    "או",
    "בין",
    "בקופת",
    "בקופות",
    "בקופה",
    "מהן",
    "מהם",
    "נדרש",
    "צריך",
    "מותר",
    "ניתן",
    "אפשר",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w\u0590-\u05FF]+", (text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _document_id(document: dict[str, Any]) -> int:
    raw_document_id = document.get("document_id")
    if raw_document_id is None:
        raw_document_id = document["id"]
    return int(raw_document_id)


def catalog_entity_ranks(question: str, documents: list[dict[str, Any]]) -> dict[int, int]:
    """Rank source-derived catalog matches for explicit law/circular references.

    This is a narrow entity channel: it only emits a document when an official
    number matches exactly or at least two normalized title terms are present in
    the query. It intentionally does not rank generic topical overlap.
    """
    query_terms = _terms(question)
    query_numbers = set(re.findall(r"\d{4}-\d{1,2}-\d{1,3}", question))
    scored: list[tuple[float, int]] = []
    for document in documents:
        title = str(document.get("canonical_title") or document.get("title") or "")
        title_terms = _terms(title)
        official_number = str(document.get("official_number") or "").replace("–", "-")
        number_match = bool(official_number and official_number in query_numbers)
        matched_title_terms = query_terms & title_terms
        if not number_match and len(matched_title_terms) < 2:
            continue
        coverage = len(matched_title_terms) / max(len(title_terms), 1)
        score = (100.0 if number_match else 0.0) + len(matched_title_terms) + coverage
        scored.append((score, _document_id(document)))
    scored.sort(reverse=True)
    return {document_id: rank for rank, (_, document_id) in enumerate(scored, 1)}


class DocumentRetriever:
    """Rank documents by profile text overlap; no embeddings, deterministic."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        document_sections: dict[int, list[str]] | None = None,
    ):
        self.documents = documents
        self.document_sections = document_sections or {}

    @staticmethod
    def _profile_text(
        document: dict[str, Any],
        sections: list[str] | None = None,
    ) -> str:
        title = str(document.get("canonical_title") or document.get("title") or "")
        topics = " ".join(document.get("topics") or [])
        outline = " ".join((document.get("heading_outline") or [])[:40])
        section_text = " ".join((sections or [])[:120])
        return " ".join(
            [
                title,
                title,  # title weighted double
                topics,
                topics,  # topics weighted double
                document.get("profile_summary") or "",
                outline,
                section_text,
                document.get("keywords") or "",
            ]
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_terms = _terms(question)
        query_numbers = set(re.findall(r"\d{4}-\d{1,2}-\d{1,3}", question))
        section_queries = re.findall(r"סעיף\s+(\d+)", question)

        scored = []
        for document in self.documents:
            raw_document_id = document.get("document_id")
            if raw_document_id is None:
                raw_document_id = document["id"]
            document_id = int(raw_document_id)
            all_sections = self.document_sections.get(document_id, [])
            profile_terms = _terms(self._profile_text(document, all_sections))
            overlap = len(query_terms & profile_terms)
            number_hits = sum(
                1
                for number in query_numbers
                if number in str(document.get("official_number") or "")
            )
            section_hits = sum(
                1
                for section in section_queries
                if any(
                    re.search(rf"\bסעיף\s*{section}\b", s) or f"סעיף{section}" in s.replace(" ", "")
                    for s in all_sections
                )
            )
            score = overlap + 30.0 * number_hits + 20.0 * section_hits
            scored.append({"document_id": document_id, "score": score, **document})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]
