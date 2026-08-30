import asyncio

from scripts.measure_challenger_hybrid import evaluate_hybrid


def test_hybrid_evaluation_embeds_each_unique_question_once():
    profiles = [
        {"document_id": 38, "title": "מסלולי השקעה", "embedding": [1.0, 0.0]},
        {"document_id": 22, "title": "העברת כספים", "embedding": [0.0, 1.0]},
    ]
    cases = [
        {"id": "age", "question": "מסלולי השקעה", "required_document_ids": [38]},
        {"id": "age-duplicate", "question": "מסלולי השקעה", "required_document_ids": [38]},
        {"id": "transfer", "question": "העברת כספים", "required_document_ids": [22]},
    ]
    calls = []

    async def embed_queries(questions):
        calls.append(list(questions))
        return [[1.0, 0.0] if "מסלולי" in q else [0.0, 1.0] for q in questions]

    result = asyncio.run(evaluate_hybrid(profiles, cases, embed_queries))

    assert calls == [["מסלולי השקעה", "העברת כספים"]]
    assert result["metrics"]["document_recall_at_5"] == 1.0
    assert result["metrics"]["all_required_documents_recall_at_5"] == 1.0
    assert result["rows"][0]["diagnostics"] == {
        "lexical_top_5": [38, 22],
        "dense_top_5": [38, 22],
        "fused_top_5": [38, 22],
    }


def test_hybrid_evaluation_requires_every_document_for_multisource_case():
    profiles = [
        {"document_id": 15, "title": "תקנות העברה", "embedding": [1.0, 0.0]},
        {"document_id": 22, "title": "חוזר העברה", "embedding": [0.8, 0.2]},
        {"document_id": 38, "title": "השקעה", "embedding": [0.0, 1.0]},
    ]
    cases = [
        {"id": "transfer", "question": "העברה", "required_document_ids": [15, 22]},
    ]

    async def embed_queries(questions):
        return [[1.0, 0.0]]

    result = asyncio.run(evaluate_hybrid(profiles, cases, embed_queries, top_k=1))

    row = result["rows"][0]
    assert row["required_recall_at_5"] == 0.5
    assert row["all_required_documents_at_5"] == 0
    assert result["metrics"]["all_required_documents_recall_at_5"] == 0.0


def test_hybrid_evaluation_uses_section_evidence_for_exact_section_query():
    profiles = [
        {"document_id": 18, "title": "חוק הפיקוח", "embedding": [0.0, 1.0]},
        {"document_id": 4, "title": "מסמך אחר", "embedding": [1.0, 0.0]},
    ]
    cases = [
        {
            "id": "section-25",
            "question": "מה קובע סעיף 25 בחוק הפיקוח?",
            "required_document_ids": [18],
        }
    ]

    async def embed_queries(questions):
        return [[1.0, 0.0]]  # dense intentionally prefers the wrong document

    result = asyncio.run(
        evaluate_hybrid(
            profiles,
            cases,
            embed_queries,
            document_sections={18: ["סעיף 25 זכויות עמית"], 4: []},
            top_k=1,
        )
    )

    assert result["rows"][0]["selected"] == [18]
