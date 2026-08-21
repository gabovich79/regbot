from services.rag_service import format_document_header, rank_hybrid_chunks


def test_hybrid_rank_prefers_current_over_superseded_when_tied():
    superseded = {
        "document_id": 1,
        "document_title": "חוזר גמל 2016-9-8 דמי ניהול",
        "document_ref": "",
        "content": "דמי ניהול לעמית",
        "validity_status": "superseded",
    }
    current = {
        "document_id": 2,
        "document_title": "חוזר גמל 2024-9-8 דמי ניהול",
        "document_ref": "",
        "content": "דמי ניהול לעמית",
        "validity_status": "current",
    }

    ranked = rank_hybrid_chunks(
        "דמי ניהול", [(0.90, superseded), (0.88, current)]
    )

    assert ranked[0]["document_id"] == 2


def test_hybrid_rank_keeps_superseded_when_explicitly_referenced():
    superseded = {
        "document_id": 1,
        "document_title": "חוזר גמל 2016-9-8 דמי ניהול",
        "document_ref": "circular-2016-9-8.pdf",
        "content": "דמי ניהול",
        "validity_status": "superseded",
    }
    unrelated = {
        "document_id": 9,
        "document_title": "מסמך אחר",
        "document_ref": "",
        "content": "לא קשור",
        "validity_status": "current",
    }

    ranked = rank_hybrid_chunks(
        "מה קובע חוזר 2016-9-8 בנושא דמי ניהול?",
        [(0.50, superseded), (0.95, unrelated)],
    )

    assert ranked[0]["document_id"] == 1


def test_hybrid_rank_ignores_missing_validity_status_for_legacy_chunks():
    # Chunks without a validity_status (e.g. in-memory fixtures) keep ranking
    # purely by relevance, so existing behaviour is unchanged.
    a = {"document_id": 1, "document_title": "א", "document_ref": "", "content": "מילה"}
    b = {"document_id": 2, "document_title": "ב", "document_ref": "", "content": "מילה"}

    ranked = rank_hybrid_chunks("מילה", [(0.9, a), (0.8, b)])

    assert ranked[0]["document_id"] == 1


def test_document_header_flags_superseded():
    header = format_document_header(
        {
            "document_title": "חוזר גמל 2016-9-8 דמי ניהול",
            "document_ref": "x.pdf",
            "effective_date": "2016-09-08",
            "validity_status": "superseded",
        }
    )

    assert "הוחלף במסמך עדכני" in header
    assert "2016-09-08" in header


def test_document_header_omits_flag_for_current():
    header = format_document_header(
        {
            "document_title": "חוזר גמל 2024-9-8",
            "document_ref": "",
            "effective_date": "2024-09-08",
            "validity_status": "current",
        }
    )

    assert "הוחלף" not in header
    assert "תוקף: 2024-09-08" in header
