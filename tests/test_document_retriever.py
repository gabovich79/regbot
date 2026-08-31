import pytest

from services.document_retriever import DocumentRetriever, catalog_entity_ranks


def test_catalog_entity_ranks_explicit_law_title_above_unrelated_documents():
    documents = [
        {
            "document_id": 18,
            "canonical_title": "חוק הפיקוח על שירותים פיננסיים (קופות גמל), התשס״ה–2005",
            "official_number": None,
        },
        {
            "document_id": 4,
            "canonical_title": "הכרעה עקרונית בדמי ניהול",
            "official_number": None,
        },
    ]

    ranks = catalog_entity_ranks(
        "מה קובע סעיף 999 בחוק הפיקוח על שירותים פיננסיים קופות גמל?",
        documents,
    )

    assert ranks[18] == 1
    assert 4 not in ranks


@pytest.fixture
def retriever():
    return DocumentRetriever(
        documents=[
            {
                "id": 38,
                "title": "רשימת מסלולי השקעה – תיקון (2024-1471)",
                "topic": "מסלולי השקעה",
                "profile_summary": "מודל השקעות ברירת מחדל תלוי גיל",
                "keywords": "מסלול השקעה גיל",
                "official_number": "2024-1471",
            },
            {
                "id": 22,
                "title": "העברת כספים בין קופות גמל - תיקון",
                "topic": "ניודים, העברות בין קופות",
                "profile_summary": "הליכי העברת כספים בין קופות גמל",
                "keywords": "העברה ניוד",
                "official_number": "2016-9-11",
            },
        ]
    )


def test_document_retriever_selects_relevant_document_first(retriever):
    results = retriever.retrieve("מהן ההוראות למסלולי השקעה תלויי גיל?", top_k=1)

    assert results[0]["document_id"] == 38


def test_document_retriever_selects_transfer_document(retriever):
    results = retriever.retrieve("איך מעבירים כספים בין קופות גמל?", top_k=1)

    assert results[0]["document_id"] == 22


def test_document_retriever_matches_official_number(retriever):
    results = retriever.retrieve("מה קובע חוזר 2016-9-11?", top_k=1)

    assert results[0]["document_id"] == 22


def test_document_retriever_returns_scores(retriever):
    results = retriever.retrieve("מסלולי השקעה", top_k=2)

    assert len(results) == 2
    assert all("score" in r for r in results)
