from scripts.build_challenger_embeddings import (
    EMBEDDING_ENCODING,
    MAX_NODE_EMBEDDING_TOKENS,
    build_nodes,
    split_node_text_for_embedding,
)


def test_build_nodes_traverses_children_when_document_root_has_no_text():
    documents = [
        {
            "doc": {"id": 18, "title": "חוק בדיקה"},
            "text": """פרק א׳: הוראות כלליות
סעיף 25 זכויות עמית
זכויות עמית אינן ניתנות להעברה, לשעבוד או לעיקול בהתאם להוראות החוק.
""",
        }
    ]

    nodes = build_nodes(documents)

    assert nodes
    assert all(node["document_id"] == 18 for node in nodes)
    assert any("זכויות עמית" in node["raw_text"] for node in nodes)


def test_split_node_text_respects_embedding_limit_without_losing_text():
    text = "מילה " * 16000

    parts = split_node_text_for_embedding(text)

    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(
        len(EMBEDDING_ENCODING.encode(part)) <= MAX_NODE_EMBEDDING_TOKENS
        for part in parts
    )
