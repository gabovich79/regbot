from services.rag_service import rank_hybrid_chunks


def test_title_field_overlap_beats_generic_content_match():
    exact = {
        "document_id": 25,
        "document_title": "חוזר גמל 2019-9-14 גילוי עלות שנתית צפויה",
        "document_ref": "",
        "content": "עמית",
    }
    generic = {
        "document_id": 1,
        "document_title": "חוזר אחר",
        "document_ref": "",
        "content": "עמית",
    }
    ranked = rank_hybrid_chunks(
        "מה נדרש לגלות לעמית לגבי העלות השנתית הצפויה?",
        [(0.95, generic), (0.50, exact)],
    )
    assert ranked[0] == exact


def test_structured_identity_keeps_same_section_from_wrong_document_out():
    exact = {
        "document_id": 18,
        "document_title": "חוק הפיקוח על שירותים פיננסיים (קופות גמל)",
        "document_ref": "",
        "section_header": "סעיף 25",
        "content": "זכויות עמית אינן ניתנות להעברה או לשעבוד.",
    }
    wrong = {
        "document_id": 14,
        "document_title": "תקנות קופות גמל",
        "document_ref": "",
        "section_header": "סעיף 25",
        "content": "נושא אחר.",
    }
    ranked = rank_hybrid_chunks(
        "מה קובע סעיף 25 לחוק הפיקוח על שירותים פיננסיים (קופות גמל)?",
        [(0.2, exact), (0.99, wrong)],
    )
    assert ranked[0] == exact
