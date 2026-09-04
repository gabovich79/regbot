from scripts.run_ingestion_batch_pilot import classify_index_readiness


def test_index_readiness_requires_embeddings_for_every_node():
    assert classify_index_readiness(
        {"status": "validated"},
        {"node_records": 4, "embedded_node_records": 4},
    ) == "ready_for_activation"
    assert classify_index_readiness(
        {"status": "validated"},
        {"node_records": 4, "embedded_node_records": 3},
    ) == "ready_for_embedding"
    assert classify_index_readiness(
        {"status": "needs_human_review"},
        {"node_records": 4, "embedded_node_records": 4},
    ) == "needs_human_review"
