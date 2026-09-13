from scripts.measure_challenger_gate import (
    promotion_decision,
    split_promotion_tuning_cases,
)


def test_promotion_tuning_cases_exclude_unverified_records_with_reasons():
    accepted, excluded = split_promotion_tuning_cases(
        [
            {"id": "verified", "review_status": "human_verified"},
            {"id": "draft", "review_status": "provisional"},
            {"id": "adversarial", "review_status": "synthetic_adversarial"},
        ]
    )

    assert [case["id"] for case in accepted] == ["verified"]
    assert excluded == [
        {"id": "draft", "reason": "review_status:provisional"},
        {"id": "adversarial", "reason": "review_status:synthetic_adversarial"},
    ]


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


def test_promotion_gate_blocks_when_verified_tuning_sample_is_too_small():
    metrics = {
        "tuning": {
            "all_required_documents_recall_at_3": 0.0,
            "all_required_documents_recall_at_5": 1.0,
        }
    }
    failed = {"heldout": [], "legacy": []}

    assert (
        promotion_decision(
            metrics,
            failed,
            enforce_tuning_top3=False,
            sufficient_verified_tuning_cases=False,
        )
        is False
    )
