from services.rag_service import force_domain_evidence_chunks


def test_force_loan_evidence_adds_percentage_and_term_chunk():
    generic = {
        "document_id": 1,
        "chunk_index": 1,
        "content": "הלוואות לגופים מוסדיים",
    }
    exact = {
        "document_id": 36,
        "chunk_index": 9,
        "content": "הלוואה מקרן השתלמות עד 50 אחוזים ולתקופה של שבע שנים",
    }

    selected = force_domain_evidence_chunks(
        "האם אפשר לקחת הלוואה מקרן השתלמות ובאיזה תנאים?",
        [generic],
        [generic, exact],
    )

    assert exact in selected


def test_force_domain_evidence_does_not_change_unrelated_questions():
    first = {"document_id": 1, "chunk_index": 1, "content": "הלוואה"}
    second = {"document_id": 2, "chunk_index": 1, "content": "מסמך אחר"}

    assert force_domain_evidence_chunks("מה קובע חוזר דמי ניהול?", [first], [first, second]) == [first]
