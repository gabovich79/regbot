import pytest

from services.section_retriever import SectionRetriever


@pytest.fixture
def retriever():
    nodes = [
        {
            "id": 1,
            "document_id": 38,
            "node_type": "section",
            "heading": "מודל השקעות ברירת מחדל",
            "raw_text": "הצטרפות למסלול השקעות ברירת מחדל תלוי גיל.",
            "page_start": 1,
        },
        {
            "id": 2,
            "document_id": 38,
            "node_type": "subsection",
            "heading": "בני 50 ומטה",
            "raw_text": "עמית שגילו 50 ומטה משויך למסלול עד 50.",
            "page_start": 1,
            "parent_id": 1,
        },
        {
            "id": 3,
            "document_id": 38,
            "node_type": "subsection",
            "heading": "בני 50 עד 60",
            "raw_text": "עמית שגילו בין 50 ל-60 משויך למסלול הביניים.",
            "page_start": 2,
            "parent_id": 1,
        },
        {
            "id": 4,
            "document_id": 22,
            "node_type": "section",
            "heading": "טיפול בבקשת העברה",
            "raw_text": "הגוף המנהל יבדוק אם ניתן לבצע העברה.",
            "page_start": 3,
        },
    ]
    return SectionRetriever(nodes)


def test_section_retriever_returns_sections_within_documents(retriever):
    results = retriever.retrieve("מהם המסלולים לבני 50 עד 60?", document_ids=[38], top_k=3)

    assert results
    assert all(r["document_id"] == 38 for r in results)
    assert any("50" in r["raw_text"] and "60" in r["raw_text"] for r in results)


def test_section_retriever_ignores_outside_documents(retriever):
    results = retriever.retrieve("בני 50 ומטה", document_ids=[22], top_k=3)

    assert results == []


def test_section_retriever_expands_to_parent(retriever):
    results = retriever.retrieve("מסלול ברירת מחדל לבני 50", document_ids=[38], top_k=1)

    assert results[0]["id"] in {1, 2}


def test_section_retriever_scores_present(retriever):
    results = retriever.retrieve("מסלולים", document_ids=[38], top_k=2)

    assert all("score" in r for r in results)
