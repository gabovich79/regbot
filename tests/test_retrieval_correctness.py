from services.rag_service import rank_hybrid_chunks


def make_chunk(document_id, title, section, content, document_type="חוק"):
    return {
        "document_id": document_id,
        "document_title": title,
        "document_ref": title,
        "section_header": section,
        "content": content,
        "document_type": document_type,
    }


def test_explicit_section_reference_beats_related_patch():
    source = make_chunk(
        18,
        "חוק הפיקוח על שירותים פיננסיים (קופות גמל), תשסה-2005",
        "סעיף 25",
        "זכויות עמית בקופת גמל אינן ניתנות להעברה או לשעבוד.",
    )
    patch = make_chunk(
        16,
        "חוק פיקוח על שירותים פיננסיים קופות גמל 2015",
        "קטע 1",
        "תיקון סעיף 25(א)(1) והוראות תחילה.",
    )

    ranked = rank_hybrid_chunks(
        "מה קובע סעיף 25 לחוק הפיקוח על שירותים פיננסיים (קופות גמל)?",
        [(0.2, source), (0.99, patch)],
    )

    assert ranked[0] == source


def test_natural_withdrawal_question_beats_lexical_match_on_word_fund():
    source = make_chunk(
        37,
        "פקודת מס הכנסה",
        "סעיף 9(16א)",
        "כללי משיכה מקרן השתלמות בפטור ממס לאחר 6 שנים.",
    )
    distractor = make_chunk(
        36,
        "כללי השקעה החלים על גופים מוסדיים",
        "סעיף 7(א)",
        "השקעה בקרנות ובשותפויות השקעה.",
        "חוזר",
    )

    ranked = rank_hybrid_chunks(
        "מה כללי משיכה של קרן השתלמות?",
        [(0.3, source), (0.99, distractor)],
    )

    assert ranked[0] == source



def test_natural_loan_question_beats_tax_document():
    source = make_chunk(
        36,
        "כללי השקעה החלים על גופים מוסדיים 2016-9-17",
        "סעיף 8(ד)",
        "הלוואה לעמית כנגד כספים לא נזילים בקרן השתלמות עד 50 אחוזים.",
        "חוזר",
    )
    distractor = make_chunk(
        37,
        "פקודת מס הכנסה",
        "סעיף 9(16א)",
        "קרן השתלמות ומשיכת כספים בפטור ממס.",
    )

    ranked = rank_hybrid_chunks(
        "האם אפשר לקחת הלוואה מקרן השתלמות ובאיזה תנאים?",
        [(0.1, source), (0.99, distractor)],
    )

    assert ranked[0] == source


def test_unrelated_same_section_number_does_not_qualify_as_source():
    source = make_chunk(
        18,
        "חוק הפיקוח על שירותים פיננסיים (קופות גמל)",
        "סעיף 25",
        "תוכן סעיף 25.",
    )
    unrelated = make_chunk(14, "תקנות קופות גמל", "סעיף 25", "תוכן אחר.", "תקנות")

    ranked = rank_hybrid_chunks(
        "מה קובע סעיף 25 לחוק הפיקוח על שירותים פיננסיים (קופות גמל)?",
        [(0.2, source), (0.99, unrelated)],
    )

    assert ranked[0] == source




def test_max_per_document_is_preserved():
    first = make_chunk(1, "חוק א", "סעיף 1", "א")
    second = make_chunk(1, "חוק א", "סעיף 2", "ב")
    other = make_chunk(2, "חוזר ב", "סעיף 1", "ג", "חוזר")

    ranked = rank_hybrid_chunks("מה קובע?", [(0.9, first), (0.8, second), (0.7, other)], max_per_document=1)

    assert [chunk["document_id"] for chunk in ranked] == [1, 2]
