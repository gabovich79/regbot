from services.rag_service import _lexical_score, rank_hybrid_chunks


def test_lexical_score_boosts_exact_section_and_percentage_terms():
    exact = {
        "document_id": 36,
        "document_title": "כללי השקעה החלים על גופים מוסדיים",
        "document_ref": "",
        "section_header": "סעיף 8(ד) הלוואות",
        "content": "הלוואה כנגד כספים לא נזילים בקרן השתלמות עד 50% ועד שבע שנים",
    }
    generic = {
        "document_id": 37,
        "document_title": "מסמך הלוואות כללי",
        "document_ref": "",
        "section_header": "קטע 7",
        "content": "הלוואות לגופים מוסדיים",
    }

    question = "הלוואה מקרן השתלמות: סעיף 8(ד), 50%, 80%, שבע שנים"

    assert _lexical_score(question, exact) > _lexical_score(question, generic)


def test_hybrid_rank_promotes_exact_loan_rule_chunk():
    exact = {
        "document_id": 36,
        "document_title": "כללי השקעה החלים על גופים מוסדיים",
        "document_ref": "",
        "section_header": "סעיף 8(ד) הלוואות",
        "content": "הלוואה כנגד כספים לא נזילים בקרן השתלמות עד 50% ועד שבע שנים",
    }
    generic = {
        "document_id": 37,
        "document_title": "מסמך הלוואות כללי",
        "document_ref": "",
        "section_header": "קטע 7",
        "content": "הלוואות לגופים מוסדיים",
    }

    ranked = rank_hybrid_chunks(
        "הלוואה מקרן השתלמות ובאיזה תנאים?",
        [(0.95, generic), (0.60, exact)],
    )

    assert ranked[0]["document_id"] == 36
