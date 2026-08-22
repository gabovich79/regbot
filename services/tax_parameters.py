"""Year-aware tax parameters with official-source fallback."""

from __future__ import annotations

import re
from datetime import date

import httpx

TAX_PARAMETER_KEY = "training_fund_favored_deposit_limit"
DEFAULT_TAX_YEAR = date.today().year
OFFICIAL_TAX_PARAMETER_SOURCES = {
    2026: {
        "source_title": "לוח עזר לחישוב מס הכנסה — ינואר 2026 ואילך",
        "source_url": "https://www.gov.il/BlobFolder/generalpage/income-tax-monthly-deductions-booklet/he/generalInformation_income-tax-monthly-deductions-booklet_monthly-deductions-booklet-2026.pdf",
        "source_page": "12",
    },
}


def is_tax_parameter_question(question: str) -> bool:
    """Recognize questions asking for a year-sensitive training-fund amount."""
    text = question or ""
    return "קרן השתלמות" in text and any(
        term in text for term in ("תקרה", "סכום", "הפקדה מוטבת", "הפקדה המוטבת", "עדכני", "כמה", "שנה")
    )


def extract_tax_year(question: str, default: int | None = None) -> int:
    """Extract an explicit tax year, defaulting to the current calendar year."""
    match = re.search(r"\b(20\d{2})\b", question or "")
    return int(match.group(1)) if match else (default or DEFAULT_TAX_YEAR)


def format_tax_parameter_context(parameter: dict) -> str:
    """Build a clearly labelled evidence block for a year-specific parameter."""
    year = parameter["tax_year"]
    source_type = parameter.get("source_type", "official")
    return (
        f"[[TAX-PARAM-{year}]]\n"
        f"פרמטר מס רשמי לשנת המס {year}: {parameter['value_text']} ₪ לשנה.\n"
        f"מפתח: {parameter['parameter_key']} | יחידה: {parameter['unit']}\n"
        f"מקור: {parameter['source_title']} | עמוד: {parameter.get('source_page') or 'לא זמין'}\n"
        f"URL: {parameter['source_url']}\n"
        f"סוג מקור: {source_type}\n"
        "יש לציין בתשובה את שנת המס ואת המקור.\n"
        "[[/TAX-PARAM]]"
    )


async def get_tax_parameter(parameter_key: str, tax_year: int) -> dict | None:
    """Read one stored year-specific parameter from SQLite."""
    from models.database import get_db

    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM regulatory_parameters WHERE parameter_key = ? AND tax_year = ?",
            (parameter_key, tax_year),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def upsert_tax_parameter(parameter: dict) -> None:
    """Persist a verified or officially fetched parameter."""
    from models.database import get_db

    db = await get_db()
    try:
        await db.execute(
            """
            INSERT INTO regulatory_parameters
                (parameter_key, tax_year, value_text, unit, source_title,
                 source_url, source_page, source_type, verified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(parameter_key, tax_year) DO UPDATE SET
                value_text = excluded.value_text,
                unit = excluded.unit,
                source_title = excluded.source_title,
                source_url = excluded.source_url,
                source_page = excluded.source_page,
                source_type = excluded.source_type,
                verified_at = CURRENT_TIMESTAMP
            """,
            (
                parameter["parameter_key"], parameter["tax_year"], parameter["value_text"],
                parameter["unit"], parameter["source_title"], parameter["source_url"],
                parameter.get("source_page"), parameter.get("source_type", "official"),
            ),
        )
        await db.commit()
    finally:
        await db.close()


def parse_favored_deposit_limit(text: str) -> str | None:
    """Extract the amount adjacent to 'הפקדה מוטבת' from official text."""
    normalized = re.sub(r"\s+", " ", text or "")
    patterns = (
        r"([\d,]+)\s+הפקדה מוטבת",
        r"הפקדה מוטבת\s+([\d,]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return None


async def fetch_official_tax_parameter(tax_year: int) -> dict | None:
    """Fetch and parse a curated official tax booklet when local data is absent."""
    source = OFFICIAL_TAX_PARAMETER_SOURCES.get(tax_year)
    if not source:
        return None
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(source["source_url"])
        response.raise_for_status()
    try:
        import fitz
        document = fitz.open(stream=response.content, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)
        document.close()
    except Exception:
        return None
    value = parse_favored_deposit_limit(text)
    if not value:
        return None
    parameter = {
        "parameter_key": TAX_PARAMETER_KEY,
        "tax_year": tax_year,
        "value_text": value,
        "unit": "ILS_PER_YEAR",
        **source,
        "source_type": "official_web_fallback",
    }
    await upsert_tax_parameter(parameter)
    return parameter


async def get_tax_parameter_context(question: str) -> str | None:
    """Return local parameter context, falling back only to curated official URLs."""
    if not is_tax_parameter_question(question):
        return None
    tax_year = extract_tax_year(question)
    parameter = await get_tax_parameter(TAX_PARAMETER_KEY, tax_year)
    if parameter is None:
        parameter = await fetch_official_tax_parameter(tax_year)
    return format_tax_parameter_context(parameter) if parameter else None
