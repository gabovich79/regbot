from services.claude_service import append_retrieved_sources


def test_append_retrieved_sources_includes_tax_parameter_source():
    context = """
[[TAX-PARAM-2026]]
פרמטר מס רשמי לשנת המס 2026: 20,566 ₪ לשנה.
מקור: לוח ניכויים 2026 | עמוד: 12
URL: https://www.gov.il/tax-2026.pdf
[[/TAX-PARAM]]
"""

    answer = append_retrieved_sources("הסכום הוא 20,566 ₪ לשנה.", context)

    assert "TAX-PARAM-2026" in answer
    assert "https://www.gov.il/tax-2026.pdf" in answer
