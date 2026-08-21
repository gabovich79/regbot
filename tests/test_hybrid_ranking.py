from services.rag_service import rank_hybrid_chunks


def test_hybrid_rank_promotes_exact_circular_title_over_semantic_near_match():
    chunks = [
        {
            "document_id": 20,
            "document_title": "חוזר גמל 2020-9-3 מבנה אחיד העברת מידע",
            "document_ref": "regulation_h_2020-9-3.pdf",
            "content": "הוראות להעברת מידע בין גופים מוסדיים",
        },
        {
            "document_id": 28,
            "document_title": "חוזר גמל 2022-9-3 ניוד בין קופות",
            "document_ref": "notice-2022-9-3.pdf",
            "content": "הוראות ניוד כספים בין קופות גמל",
        },
    ]

    ranked = rank_hybrid_chunks(
        "מה קובע חוזר גמל 2022-9-3 בנושא ניוד בין קופות?",
        [(0.95, chunks[0]), (0.62, chunks[1])],
    )

    assert ranked[0]["document_id"] == 28


def test_hybrid_rank_limits_repeated_chunks_from_one_document():
    chunks = [
        {"document_id": 1, "document_title": "חוק א", "document_ref": "", "content": "מילה"},
        {"document_id": 1, "document_title": "חוק א", "document_ref": "", "content": "מילה"},
        {"document_id": 2, "document_title": "חוזר ב", "document_ref": "", "content": "מילה"},
    ]

    ranked = rank_hybrid_chunks(
        "מילה", [(0.9, chunks[0]), (0.8, chunks[1]), (0.7, chunks[2])], max_per_document=1
    )

    assert [chunk["document_id"] for chunk in ranked] == [1, 2]
