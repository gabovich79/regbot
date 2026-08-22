import pytest

from models import database
from services.tax_parameters import (
    TAX_PARAMETER_KEY,
    format_tax_parameter_context,
    is_tax_parameter_question,
    extract_tax_year,
    get_tax_parameter,
)


def test_tax_parameter_question_detection_and_year_extraction():
    question = "מה תקרת ההפקדה המוטבת בקרן השתלמות לעצמאי בשנת 2026?"
    assert is_tax_parameter_question(question) is True
    assert extract_tax_year(question) == 2026
    assert extract_tax_year("מה הסכום העדכני?") == 2026


def test_non_parameter_question_is_not_classified():
    assert is_tax_parameter_question("איך מעבירים קרן השתלמות?") is False


@pytest.mark.asyncio
async def test_seeded_tax_parameter_has_official_source(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "regbot.db"))
    await database.init_db()

    parameter = await get_tax_parameter(TAX_PARAMETER_KEY, 2026)

    assert parameter["value_text"] == "20,566"
    assert parameter["source_type"] == "official"
    assert "gov.il" in parameter["source_url"]
    assert parameter["tax_year"] == 2026


def test_tax_parameter_context_requires_year_and_source():
    context = format_tax_parameter_context({
        "parameter_key": TAX_PARAMETER_KEY,
        "tax_year": 2026,
        "value_text": "20,566",
        "unit": "ILS_PER_YEAR",
        "source_title": "חוברת ניכויים חודשית 2026 — רשות המסים",
        "source_url": "https://www.gov.il/example.pdf",
        "source_page": "12",
        "source_type": "official",
    })

    assert "20,566" in context
    assert "2026" in context
    assert "https://www.gov.il/example.pdf" in context
    assert "TAX-PARAM-2026" in context
