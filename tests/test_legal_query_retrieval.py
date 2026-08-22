from services.rag_service import build_retrieval_queries, rank_hybrid_chunks


def test_build_retrieval_queries_decomposes_transfer_pledge_garnishment_question():
    question = "העברה, שעבוד או עיקול של זכויות עמית?"

    queries = build_retrieval_queries(question)

    assert queries[0] == question
    assert any("שעבוד" in query and "זכויות עמית" in query for query in queries)
    assert any("עיקול" in query and "זכויות עמית" in query for query in queries)
    assert any("העברה" in query and "זכויות עמית" in query for query in queries)


def test_build_retrieval_queries_keeps_simple_question_single():
    assert build_retrieval_queries("מהו חוזר 2016-9-11?") == ["מהו חוזר 2016-9-11?"]


def test_authority_ranking_promotes_law_for_rights_question():
    personal_fund_rules = {
        "document_id": 13,
        "document_title": "תקנות קופת גמל בניהול אישי — 2009",
        "document_ref": "",
        "content": "הוראות כלליות לניהול קופה",
        "document_type": "תקנה",
    }
    provident_fund_law = {
        "document_id": 18,
        "document_title": "חוק הפיקוח על שירותים פיננסיים (קופות גמל), תשסה-2005",
        "document_ref": "",
        "content": "איסור העברה, שעבוד או עיקול של זכויות עמית",
        "document_type": "חוק",
    }

    ranked = rank_hybrid_chunks(
        "העברה, שעבוד או עיקול של זכויות עמית",
        [(0.95, personal_fund_rules), (0.70, provident_fund_law)],
    )

    assert ranked[0]["document_id"] == 18
