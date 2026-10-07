from copy import deepcopy

import pytest

from services.acceptance import release_gate


def reports():
    cases = [{'id': str(i), 'gold_ready': True, 'passed': True, 'critical_error': False,
              'required_units': 2, 'retrieved_units': 2, 'response_time_ms': 1000,
              'checks': {k: True for k in ('correct', 'complete', 'supported', 'handles_missing', 'handles_conflicts')}} for i in range(20)]
    return [{'run_id': str(i), 'runtime_fingerprint': 'runtime', 'split': 'acceptance',
             'case_fingerprint': 'frozen', 'cases': deepcopy(cases)} for i in range(3)]


@pytest.mark.parametrize('value', [None, '1000', float('nan'), float('inf'), True, -1, 0, 90001])
def test_invalid_latency_is_a_blocker_not_a_crash(value):
    runs = reports()
    runs[0]['cases'][0]['response_time_ms'] = value
    result = release_gate(runs, human_approved=True)
    assert not result['release_ready']
    assert 'invalid_or_exceeded_deadline' in result['blockers']


@pytest.mark.parametrize('changes', [
    {'retrieved_units': 3}, {'required_units': -1}, {'required_units': True},
    {'passed': 'true'}, {'critical_error': True}, {'gold_ready': False},
    {'checks': {'handles_missing': True, 'handles_conflicts': True}},
])
def test_invalid_or_inconsistent_case_cannot_pass_gate(changes):
    runs = reports()
    runs[0]['cases'][0].update(changes)
    assert release_gate(runs, True) == {'release_ready': False, 'blockers': ['invalid_run_report']}


@pytest.mark.parametrize('runs', [None, [None], [{'run_id': []}], [{'cases': [None]}]])
def test_corrupt_report_fails_closed(runs):
    assert not release_gate(runs, True)['release_ready']
