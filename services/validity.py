"""Document validity helpers for regulatory corpus curation.

A regulatory document can be:
  - current: the latest word on a subject,
  - superseded: replaced by a newer document (``superseded_by`` points at it),
  - expired: past its ``valid_until`` date with no replacement.

None of these states removes a document from retrieval: historical questions
still need superseded/expired text. Validity is a ranking signal and a
transparency flag, not a hard filter.
"""

from __future__ import annotations

import re
from datetime import date

# Year-first dates are the convention in Israeli circular titles
# (e.g. "חוזר גמל 2024-9-8"). Deliberately NOT parsing day-first D.M.YYYY to
# avoid month/day ambiguity.
_YEAR_FIRST_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")


def extract_date_from_title(title: str | None) -> str | None:
    """Return an ISO date (YYYY-MM-DD) from a circular title, or None."""
    if not title:
        return None
    match = _YEAR_FIRST_DATE.search(title)
    if not match:
        return None
    year, month, day = (int(group) for group in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def document_validity_status(document: dict, today: str | None = None) -> str:
    """Classify a document as current, superseded, or expired."""
    if document.get("superseded_by") is not None:
        return "superseded"
    valid_until = document.get("valid_until")
    if valid_until:
        reference = today or date.today().isoformat()
        if valid_until < reference:
            return "expired"
    return "current"
