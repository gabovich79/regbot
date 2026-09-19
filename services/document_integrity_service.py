"""Source-first integrity checks for document identity metadata."""

from __future__ import annotations

import re
from typing import Any


OFFICIAL_NUMBER_PATTERN = re.compile(
    r"(?<![0-9-])(?:19|20)\d{2}[-–]\d{1,2}[-–]\d{1,3}(?![0-9-])"
)
GENERIC_TERMS = {
    "חוזר",
    "גמל",
    "קופות",
    "קופת",
    "מסמך",
    "תיקון",
    "כללי",
    "הוראות",
    "מדינת",
    "ישראל",
    "רשות",
    "שוק",
    "ההון",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w\u0590-\u05FF]+", (text or "").lower())
        if len(token) > 2
        and not token.isdigit()
        and token not in GENERIC_TERMS
        and token not in {"pdf", "docx", "doc"}
    }


def _normalize_title(text: str) -> str:
    normalized = re.sub(r"[_.]+", " ", text or "")
    normalized = re.sub(r"\.(?:pdf|docx|doc)\b", "", normalized)
    return normalized


def _official_numbers(text: str) -> set[str]:
    return {
        value.replace("–", "-")
        for value in OFFICIAL_NUMBER_PATTERN.findall(text or "")
    }


def _detect_ocr_reversal(text: str) -> bool:
    """Flag extracted text whose Hebrew words are visually reversed (RTL misread)."""
    sample = (text or "")[:4000]
    words = [word for word in re.findall(r"[\u0590-\u05FF]+", sample) if len(word) >= 3]
    if len(words) < 4:
        return False
    known_reversed = {
        "לכ": "כל",
        "של": "לש",
        "חוטיבל": "לביטוח",
        "הרבח": "חברה",
        "הםיאולימ": "מילואים",
        "תודקפה": "הפקדות",
        "םיפסכ": "כספים",
        "תופוקו": "קופות",
        "היסנפ": "פנסיה",
        "תונרק": "קרנות",
        "תפקמ": "מקפת",
        "וקופות": "תופוקו",
    }
    matches = sum(word in known_reversed for word in words[:40])
    return matches >= 3 or (len(words) >= 4 and matches / len(words) >= 0.3)


def _detect_binary_text(text: str) -> bool:
    """Flag extracted text that is mostly binary/garbled rather than Hebrew prose."""
    sample = (text or "")[:8000]
    if not sample:
        return False
    null_ratio = sample.count("\x00") / max(len(sample), 1)
    return null_ratio > 0.05


def assess_document_integrity(
    document: dict[str, Any], text: str, profile: dict[str, Any]
) -> dict[str, Any]:
    """Return a conservative integrity status; never mutate curator metadata."""
    reasons: list[str] = []
    if not (text or "").strip():
        return {"status": "failed", "reasons": ["empty_extracted_text"]}

    if _detect_ocr_reversal(text):
        reasons.append("extraction_hebrew_reversed")
    if _detect_binary_text(text):
        reasons.append("extraction_binary")

    title = _normalize_title(str(document.get("title") or ""))
    title_numbers = _official_numbers(title)
    content_numbers = _official_numbers((text or "")[:20000])
    if title_numbers and content_numbers and title_numbers.isdisjoint(content_numbers):
        reasons.append("official_number_conflict")

    title_terms = _terms(title)
    body_terms = _terms((text or "")[:30000])
    distinctive_title_terms = title_terms - GENERIC_TERMS
    if len(distinctive_title_terms) >= 2:
        overlap = len(distinctive_title_terms & body_terms) / len(distinctive_title_terms)
        if overlap < 0.35:
            reasons.append("title_body_mismatch")

    configured_topic = str(document.get("topic") or "").strip()
    topic_terms = _terms(configured_topic)
    if topic_terms and len(topic_terms & body_terms) / len(topic_terms) < 0.25:
        reasons.append("topic_body_mismatch")

    if not profile.get("identity_evidence"):
        reasons.append("identity_evidence_missing")

    if "official_number_conflict" in reasons or "empty_extracted_text" in reasons:
        status = "failed"
    elif reasons:
        status = "warning"
    else:
        status = "verified"
    return {"status": status, "reasons": reasons}
