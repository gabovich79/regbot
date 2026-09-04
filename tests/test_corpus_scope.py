from scripts.corpus_scope import annotate_case_blocked, load_active_document_ids


def test_active_ids_exclude_approved_decisions():
    active = load_active_document_ids()

    assert "8" not in active
    assert "9" not in active
    assert "4" not in active
    assert "16" not in active
    assert "25" not in active
    assert "29" not in active
    assert "17" in active
    assert "18" in active


def test_case_requiring_inactive_document_is_blocked():
    active = load_active_document_ids()

    case = annotate_case_blocked(
        {"id": "heldout-ruling-52a", "required_document_ids": [4]},
        active,
    )

    assert "blocked_reason" in case
    assert "4" in case["blocked_reason"]


def test_case_requiring_only_active_documents_is_not_blocked():
    active = load_active_document_ids()

    case = annotate_case_blocked(
        {"id": "tuning-case", "required_document_ids": [18, 22]},
        active,
    )

    assert "blocked_reason" not in case
