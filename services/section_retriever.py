"""Section-level retriever scoped to selected documents, with parent expansion."""

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


class SectionRetriever:
    """Rank nodes inside selected documents; expand leaf hits to parent section."""

    def __init__(self, nodes: list[dict[str, Any]]):
        self.nodes = {int(node["id"]): node for node in nodes}

    def _node_text(self, node: dict[str, Any]) -> str:
        return " ".join(
            str(node.get(field) or "")
            for field in ("heading", "section_label", "raw_text")
        )

    def retrieve(
        self, question: str, document_ids: list[int], top_k: int = 5
    ) -> list[dict[str, Any]]:
        allowed = set(document_ids)
        query_terms = _terms(question)

        ranked = []
        for node in self.nodes.values():
            if node["document_id"] not in allowed:
                continue
            text_terms = _terms(self._node_text(node))
            overlap = len(query_terms & text_terms)
            if overlap:
                ranked.append((overlap, node))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [node for _, node in ranked[:top_k]]

        # Parent expansion: keep the parent section in context when a leaf hit.
        expanded: dict[int, dict[str, Any]] = {}
        for node in selected:
            expanded[int(node["id"])] = node
            parent_id = node.get("parent_id")
            if parent_id and int(parent_id) in self.nodes:
                expanded[int(parent_id)] = self.nodes[int(parent_id)]
        return [
            {**node, "score": float(overlap)}
            for overlap, node in ranked[: top_k + len(expanded)]
            if node["id"] in expanded
        ]
