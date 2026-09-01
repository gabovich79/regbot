from scripts.measure_challenger_gate import promotion_decision


def test_promotion_gate_blocks_legacy_failure():
    metrics = {
        "tuning": {
            "all_required_documents_recall_at_3": 0.857,
            "all_required_documents_recall_at_5": 1.0,
        }
    }
    failed = {"heldout": [], "legacy": ["fund-mobility"]}

    assert promotion_decision(metrics, failed) is False


def test_promotion_gate_accepts_clean_recall_gate():
    metrics = {
        "tuning": {
            "all_required_documents_recall_at_3": 0.857,
            "all_required_documents_recall_at_5": 1.0,
        }
    }
    failed = {"heldout": [], "legacy": []}

    assert promotion_decision(metrics, failed) is True
