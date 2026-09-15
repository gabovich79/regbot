import pytest

from services.local_evidence_index import build_local_evidence_index, search_local_evidence_index


def test_local_evidence_index_returns_the_semantically_closest_evidence_unit():
    units = [
        {"evidence_id": "D37-9", "raw_text": "קרן השתלמות פטור לאחר שש שנים"},
        {"evidence_id": "D37-125", "raw_text": "מס על ריבית והפניה לסעיף 121"},
    ]
    index = build_local_evidence_index(
        units,
        model="nomic-embed-text:v1.5",
        embed_many=lambda texts: [[1.0, 0.0], [0.0, 1.0]],
    )

    hits = search_local_evidence_index(
        "מה המס על ריבית?",
        index,
        embed_query=lambda question: [0.1, 0.9],
        top_k=1,
    )

    assert [hit["evidence_id"] for hit in hits] == ["D37-125"]
    assert hits[0]["score"] > 0.9


def test_local_evidence_index_rejects_missing_or_dimension_mismatched_vectors():
    with pytest.raises(ValueError, match="one embedding per evidence unit"):
        build_local_evidence_index(
            [{"evidence_id": "D37-9", "raw_text": "ראיה"}],
            model="nomic-embed-text:v1.5",
            embed_many=lambda texts: [],
        )

    with pytest.raises(ValueError, match="dimension"):
        build_local_evidence_index(
            [
                {"evidence_id": "D37-9", "raw_text": "ראיה א"},
                {"evidence_id": "D37-125", "raw_text": "ראיה ב"},
            ],
            model="nomic-embed-text:v1.5",
            embed_many=lambda texts: [[1.0], [0.0, 1.0]],
        )
