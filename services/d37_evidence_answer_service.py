"""Evidence-first prompt, deterministic citation resolution, and gate for D37."""

from __future__ import annotations

import re
from typing import Any

_CITATION_ID_RE = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]")
_HIGH_CONFIDENCE_MARKERS = ("CONFIDENCE HIGH", "ביטחון גבוה")


def build_grounded_prompt(question: str, evidence_units: list[dict[str, Any]]) -> str:
    """Build a closed-book Hebrew prompt containing only retrieved source evidence."""
    blocks = []
    for unit in evidence_units:
        blocks.append(
            "\n".join(
                [
                    f"[[{unit['evidence_id']}]]",
                    f"מסמך: {unit.get('source_id', 'לא צוין')} | סעיף: {unit['section']} | "
                    f"עמודים: {unit['page_start']}-{unit['page_end']}",
                    str(unit["raw_text"]),
                    f"[[/{unit['evidence_id']}]]",
                ]
            )
        )
    evidence = "\n\n".join(blocks)
    return f"""ענה בעברית בלבד על השאלה, אך ורק על סמך הראיות הסגורות בהמשך.
אין להשתמש בהקשר חיצוני, בידע קודם או בחישוב שאינו מופיע בראיות.
לכל טענה משפטית הוסף סימון מקור בלבד, בפורמט [[evidence_id]] — אין להעתיק את הטקסט, רק לציין את מזהה הראיה.
אם הראיות אינן מספיקות לפרט מבוקש, אמור זאת במפורש והסבר איזה מקור מורשה חסר.
אסור לכתוב CONFIDENCE HIGH או לטעון לוודאות מעבר לראיות.

שאלה:
{question}

ראיות:
{evidence}
"""


def resolve_citations(answer: str, evidence_units: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach the exact verbatim source span to each citation pointer in the answer.

    The generation model only writes ``[[evidence_id]]`` pointers. This function is the
    deterministic layer that pulls the verbatim text from the accepted evidence units,
    so the final citation can never drift from the source.
    """
    evidence_by_id = {str(unit["evidence_id"]): unit for unit in evidence_units}
    seen: list[str] = []
    citations: list[dict[str, Any]] = []
    unresolved_ids: list[str] = []
    for citation_id in _CITATION_ID_RE.findall(answer):
        if citation_id in seen:
            continue
        seen.append(citation_id)
        unit = evidence_by_id.get(citation_id)
        if unit is None:
            unresolved_ids.append(citation_id)
            continue
        citations.append(
            {
                "evidence_id": citation_id,
                "section": unit.get("section"),
                "page_start": unit.get("page_start"),
                "page_end": unit.get("page_end"),
                "verbatim": unit.get("raw_text"),
            }
        )
    return {
        "answer": answer,
        "citations": citations,
        "unresolved_ids": unresolved_ids,
    }


def gate_grounded_answer(answer: str, evidence_units: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject answers that cite unavailable evidence or assert impermissible confidence.

    Verbatim transcription is no longer required from the model: the resolver attaches the
    exact span, and entailment is scored separately. This gate only enforces the
    deterministic invariants — every pointer resolves to a retrieved unit, at least one
    citation exists, and confidence stays capped while evidence is not human-verified.
    """
    allowed_ids = {str(unit["evidence_id"]) for unit in evidence_units}
    citation_ids: list[str] = []
    for citation_id in _CITATION_ID_RE.findall(answer):
        if citation_id not in citation_ids:
            citation_ids.append(citation_id)
    unknown = [citation_id for citation_id in citation_ids if citation_id not in allowed_ids]

    reasons: list[str] = []
    if not citation_ids:
        reasons.append("missing_citation")
    if unknown:
        reasons.append("unknown_citation")
    normalized_answer = answer.upper()
    if any(marker.upper() in normalized_answer for marker in _HIGH_CONFIDENCE_MARKERS):
        reasons.append("high_confidence")

    pending_human_review = any(
        unit.get("verification_status") != "human_verified" for unit in evidence_units
    )
    return {
        "status": "reject" if reasons else "pass",
        "reasons": reasons,
        "citation_ids": citation_ids,
        "unknown_citation_ids": unknown,
        "confidence": (
            "not_high_pending_human_review"
            if pending_human_review
            else "human_verified_evidence_available"
        ),
    }
