from services.legal_rules import get_training_fund_loan_rule_context


def test_training_fund_loan_rule_context_contains_regulatory_limits():
    context = get_training_fund_loan_rule_context("האם אפשר לקחת הלוואה מקרן השתלמות?")

    assert "80%" in context
    assert "50%" in context
    assert "7 שנים" in context
    assert "2016-9-17" in context


def test_training_fund_loan_rule_context_ignores_unrelated_question():
    assert get_training_fund_loan_rule_context("מה תנאי משיכת קרן השתלמות?") is None
