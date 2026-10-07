"""Necessary literal calendar-date support, never a legal applicability verdict."""
from datetime import date
import re

MONTHS = ('ינואר','פברואר','מרץ','אפריל','מאי','יוני','יולי','אוגוסט','ספטמבר','אוקטובר','נובמבר','דצמבר')


def date_present(iso, text):
    value = date.fromisoformat(iso)
    normalized = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    normalized = re.sub(r'\s+', ' ', normalized)
    day, month, year = value.day, value.month, value.year
    # Day-first conventional dates cannot be confused with year-first circular IDs.
    if re.search(rf'(?<!\d)0?{day}[./]0?{month}[./]{year}(?!\d)', normalized):
        return True
    word = MONTHS[month-1]
    if re.search(rf'(?<!\d){day}\s+ב?{word}\s+{year}(?!\d)',normalized):
        return True
    # PDF text can store the two numeric tokens in visual RTL order.
    if re.search(rf'(?<!\d){year}\s+ב?{word}\s+{day}(?!\d)',normalized):
        return True
    # ISO dates require an explicit temporal cue; a bare regulatory identifier
    # such as 2021-9-5 or "circular 2021-09-05" is not date evidence.
    return bool(re.search(rf'(?:effective(?: from)?|valid from|תחילה|בתוקף|מיום)\s*:?\s*{re.escape(iso)}(?!\d)',normalized,re.I))
