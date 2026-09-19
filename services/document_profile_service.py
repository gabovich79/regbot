"""Build retrieval-only document profiles from source-derived identity text."""

from __future__ import annotations

import json
import re
from typing import Any


OFFICIAL_NUMBER_PATTERN = re.compile(
    r"(?<![0-9-])(?:19|20)\d{2}[-–]\d{1,2}[-–]\d{1,3}(?![0-9-])"
)
GENERIC_IDENTITY_LINES = (
    "מדינת ישראל",
    "רשות שוק ההון",
    "אגף שוק ההון",
    "משרד האוצר",
)


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in (text or "").splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]


def _strip_file_extension(title: str) -> str:
    """Remove a trailing upload filename extension so identity can use the title."""
    return re.sub(r"\.(?:pdf|docx?|txt)$", "", title.strip(), flags=re.IGNORECASE)


def _extract_official_number(title: str, text: str) -> str | None:
    sample = text or ""
    candidates: list[tuple[int, str, bool]] = []
    for match in OFFICIAL_NUMBER_PATTERN.finditer(sample[:12000]):
        number = match.group(0).replace("–", "-")
        before = sample[max(0, match.start() - 80):match.start()]
        if re.search(
            r"(?:חוזר(?:\s+גופים\s+מוסדיים)?|תקנות(?:\s+הפיקוח)?|חוק|פקודת|הוראות)\s*$",
            before,
        ):
            after = sample[match.end():match.end() + 60]
            is_header = bool(re.search(r"סיווג|כללי|מבנה|הסדרת|עדכון|תיקון", after))
            candidates.append((match.start(), number, is_header))
    if candidates:
        header_candidates = [candidate for candidate in candidates if candidate[2]]
        selected = header_candidates if header_candidates else candidates
        return min(selected, key=lambda item: item[0])[1]
    content_matches = OFFICIAL_NUMBER_PATTERN.findall(sample[:12000])
    if content_matches:
        return content_matches[0].replace("–", "-")
    title_matches = OFFICIAL_NUMBER_PATTERN.findall(title or "")
    return title_matches[0].replace("–", "-") if title_matches else None


def _looks_like_identity_title(line: str) -> bool:
    if len(line) < 5 or len(line) > 220:
        return False
    if line[0] in "][()\"'«»“”":
        return False
    if any(line.startswith(prefix) for prefix in GENERIC_IDENTITY_LINES):
        return False
    if line.startswith(("סיווג", "בתוקף סמכות", "תוכן", "תאריך")):
        return False
    if re.fullmatch(r"[\d\s./-]+", line):
        return False
    return any(
        term in line
        for term in (
            "חוק",
            "תקנות",
            "חוזר",
            "העברת",
            "מסלולי",
            "פקודת",
            "הוראות",
            "רשימת",
            "תיקון",
            "השקעה",
            "הלווא",
            "הצטרפות",
            "עלות",
            "דמי",
            "ניוד",
            "מבנה",
            "דוחות",
            "אימות",
            "תמותה",
            "מילואים",
            "משיכ",
        )
    )


def _extract_canonical_title(stored_title: str, text: str, official_number: str | None) -> str:
    lines = _clean_lines(text)[:40]
    stored = _strip_file_extension(stored_title).strip()
    # A curator-chosen stored title is authoritative unless it is a raw
    # filename; integrity checks later compare it against the body text.
    looks_like_filename = (
        "_" in stored
        or stored.lower().endswith((".pdf", ".docx", ".doc"))
        or len(stored) < 5
    )
    if stored and not looks_like_filename and _looks_like_identity_title(stored):
        return stored
    if official_number:
        normalized_number = official_number.replace("-", "[-–]")
        for index, line in enumerate(lines):
            if re.search(normalized_number, line):
                for candidate in lines[index + 1:index + 5]:
                    if _looks_like_identity_title(candidate):
                        return candidate
    for line in lines:
        if _looks_like_identity_title(line) and not OFFICIAL_NUMBER_PATTERN.search(line):
            return line
    if stored and _looks_like_identity_title(stored):
        return stored
    return stored or (lines[0] if lines else "")


def _detect_issuer(text: str) -> str | None:
    sample = (text or "")[:12000]
    if "רשות שוק ההון" in sample or "אגף שוק ההון" in sample:
        return "רשות שוק ההון, ביטוח וחיסכון"
    if "רשות המסים" in sample or "מס הכנסה" in sample:
        return "רשות המסים בישראל"
    if "הכנסת" in sample:
        return "מדינת ישראל"
    return None


def _detect_document_type(title: str, text: str, configured: str | None) -> str | None:
    if configured:
        return configured
    sample = f"{title}\n{text[:3000]}"
    if "פקודת" in sample or "חוק" in sample:
        return "חוק"
    if "תקנות" in sample:
        return "תקנות"
    if "חוזר" in sample:
        return "חוזר"
    if "טיוטה" in sample:
        return "טיוטה"
    return None


def _extract_identity_evidence(text: str, canonical_title: str, official_number: str | None) -> list[str]:
    lines = _clean_lines(text)[:50]
    evidence: list[str] = []
    for line in lines:
        if canonical_title in line or line in canonical_title:
            evidence.append(line)
        elif official_number and official_number in line.replace("–", "-"):
            evidence.append(line)
        elif _looks_like_identity_title(line) and len(line) <= 120:
            evidence.append(line)
        if len(evidence) == 3:
            break
    return list(dict.fromkeys(evidence))


def _extract_summary(text: str, identity_evidence: list[str]) -> str:
    lines = _clean_lines(text)
    evidence = set(identity_evidence)
    candidates = [
        line
        for line in lines
        if line not in evidence
        and len(line) >= 30
        and not OFFICIAL_NUMBER_PATTERN.fullmatch(line)
        and not any(line.startswith(prefix) for prefix in GENERIC_IDENTITY_LINES)
    ]
    preferred = next(
        (line for line in candidates if "מטרת" in line or "מסדיר" in line),
        candidates[0] if candidates else "",
    )
    return preferred[:300]


def _extract_outline(text: str) -> list[str]:
    lines = _clean_lines(text)
    headings = []
    for line in lines:
        if len(line) > 140:
            continue
        if line.endswith((".", ":", ";", ",")):
            continue
        if re.match(r"^(?:פרק|סעיף|תקנה)\s+", line) or line in {
            "כללי",
            "הגדרות",
            "תחולה",
            "תחילה",
            "ביטול חוזרים",
            "מסירת נתונים",
        }:
            headings.append(line)
        elif 8 <= len(line) <= 60 and not any(
            marker in line for marker in ("- ", " – ")
        ):
            headings.append(line)
        if len(headings) == 20:
            break
    return list(dict.fromkeys(headings))


def build_document_profile(document: dict[str, Any], text: str) -> dict[str, Any]:
    """Create a source-derived profile; summaries are retrieval metadata, not evidence."""
    stored_title = str(document.get("title") or "").strip()
    official_number = _extract_official_number(stored_title, text)
    canonical_title = _extract_canonical_title(stored_title, text, official_number)
    identity_evidence = _extract_identity_evidence(text, canonical_title, official_number)
    profile_summary = _extract_summary(text, identity_evidence)
    configured_topic = str(document.get("topic") or "").strip()
    topics = [configured_topic] if configured_topic else [canonical_title]

    return {
        "document_id": int(document["id"]),
        "canonical_title": canonical_title,
        "official_number": official_number,
        "issuer": _detect_issuer(text),
        "document_type": _detect_document_type(
            canonical_title, text, document.get("document_type")
        ),
        "publication_date": None,
        "effective_date": document.get("effective_date"),
        "valid_until": document.get("valid_until"),
        "lifecycle_status": document.get("lifecycle_status") or "current",
        "supersedes_document_id": document.get("superseded_by"),
        "profile_summary": profile_summary,
        "scope_in": [profile_summary] if profile_summary else [],
        "scope_out": [],
        "topics": topics,
        "keywords": [],
        "heading_outline": _extract_outline(text),
        "identity_evidence": identity_evidence,
        "profile_embedding": None,
        "review_status": "machine",
        "profile_version": 1,
    }
