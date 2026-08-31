import asyncio

from scripts.run_challenger_ablations import run_ablations


def test_run_ablations_emits_one_labeled_result_per_configuration():
    profiles = [
        {"document_id": 18, "title": "חוק הפיקוח", "embedding": [1.0, 0.0]},
        {"document_id": 4, "title": "מסמך אחר", "embedding": [0.0, 1.0]},
    ]
    cases = [
        {
            "id": "section-25",
            "question": "מה קובע סעיף 25?",
            "required_document_ids": [18],
        }
    ]
    configs = [
        {"id": "union-5-node-0", "candidate_depth": 5, "node_rrf_weight": 0},
        {"id": "union-5-node-1", "candidate_depth": 5, "node_rrf_weight": 1},
    ]
    calls = []

    async def embed_queries(questions):
        calls.append(questions)
        return [[1.0, 0.0]]

    result = asyncio.run(
        run_ablations(profiles, cases, embed_queries, configs=configs)
    )

    assert calls == [["מה קובע סעיף 25?"]]
    assert [row["configuration"]["id"] for row in result["runs"]] == [
        "union-5-node-0",
        "union-5-node-1",
    ]
    assert all("metrics" in row and "rows" in row for row in result["runs"])
