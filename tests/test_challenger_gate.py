from scripts.measure_challenger_gate import load_legacy_cases


def test_load_legacy_cases_maps_every_case_to_document_ids():
    cases = load_legacy_cases()

    by_id = {case["id"]: case for case in cases}
    assert len(cases) == 17
    assert by_id["fees-management"]["required_document_ids"] == [19]
    assert by_id["transfer-funds"]["required_document_ids"] == [24, 22]
    assert by_id["age-dependent-tracks"]["required_document_ids"] == [38]
    assert by_id["severance-continuity"]["required_document_ids"] == [12]
    assert all(case["required_document_ids"] for case in cases)
    assert all(case["question"] for case in cases)
