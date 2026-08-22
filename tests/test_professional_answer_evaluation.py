from services.evaluation_service import score_professional_answer


def test_professional_answer_scores_conclusion_conditions_and_actionability():
    case = {
        "id": "transfer-case",
        "expected_conclusion": ["ניתן לנייד", "בכפוף לתנאים"],
        "required_concepts": ["העברת כספים", "הקופה המקבלת"],
        "required_actions": ["לבדוק", "מסמכים"],
        "must_not_include": ["תמיד"],
        "expected_citation_prefixes": ["D20-"],
    }
    answer = (
        "ניתן לנייד בכפוף לתנאים. יש לבדוק העברת כספים מול הקופה המקבלת "
        "ולהכין מסמכים [D20-P3]."
    )
    result = score_professional_answer(case, answer)

    assert result["passed"] is True
    assert result["conclusion_score"] == 1.0
    assert result["required_concepts_score"] == 1.0
    assert result["actionability_score"] == 1.0


def test_professional_answer_reports_missing_conclusion_and_unsupported_claim():
    case = {
        "id": "withdrawal-case",
        "expected_conclusion": ["לא ניתן לקבוע"],
        "required_concepts": ["תנאים"],
        "required_actions": [],
        "must_not_include": ["פטור מלא ממס"],
    }
    result = score_professional_answer(case, "יש פטור מלא ממס.")

    assert result["passed"] is False
    assert result["conclusion_score"] == 0.0
    assert result["missing_required_concepts"] == ["תנאים"]
    assert result["prohibited_terms_found"] == ["פטור מלא ממס"]


def test_professional_answer_requires_clarification_when_case_demands_it():
    case = {
        "id": "ambiguous-case",
        "requires_clarification": True,
        "clarification_terms": ["איזה מוצר", "מתי"],
    }
    result = score_professional_answer(case, "כדי לענות צריך לדעת איזה מוצר ומתי.")

    assert result["passed"] is True
    assert result["clarification_score"] == 1.0
