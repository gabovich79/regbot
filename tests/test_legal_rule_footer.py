from services.claude_service import append_retrieved_sources


def test_append_retrieved_sources_includes_loan_rule_source():
    context = """
[[LEGAL-RULE LOAN-2016-9-17-8D]]
הלוואה מקרן השתלמות עד 50% ועד שבע שנים.
[[/LEGAL-RULE]]
"""

    answer = append_retrieved_sources("ניתן לקחת הלוואה בכפוף לתנאים.", context)

    assert "LEGAL-RULE LOAN-2016-9-17-8D" in answer
    assert "2016-9-17" in answer
