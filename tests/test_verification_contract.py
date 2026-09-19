from copy import deepcopy

import pytest

from services.evidence_pipeline import verification_result, run_pipeline


def complete_check():
    return {'checks': [{'index': 0, 'supported': True}], 'missing': [], 'conflicts': [],
            'issue_checks': [{'issue_index': 0, 'status': 'covered', 'claim_indices': [0]},
                             {'issue_index': 1, 'status': 'missing', 'claim_indices': []}]}


def test_each_question_aspect_requires_explicit_coverage():
    claims = [{'index': 0}]
    accepted, missing, conflicts, valid = verification_result(complete_check(), claims, ['rule', 'exceptions'])
    assert valid and accepted == claims and missing == ['exceptions'] and not conflicts


@pytest.mark.parametrize('mutation', [
    lambda c: c.pop('conflicts'),
    lambda c: c.update(missing='not a list'),
    lambda c: c.update(checks=[]),
    lambda c: c['checks'].append({'index': 1, 'supported': True}),
    lambda c: c['checks'].append(dict(c['checks'][0])),
    lambda c: c['checks'][0].update(index=False),
    lambda c: c.update(issue_checks=[]),
    lambda c: c['issue_checks'][1].update(issue_index=0),
    lambda c: c['issue_checks'][0].update(claim_indices=[999]),
])
def test_partial_or_malformed_verifier_cannot_approve(mutation):
    checked = complete_check()
    mutation(checked)
    accepted, missing, _, valid = verification_result(checked, [{'index': 0}], ['rule', 'exceptions'])
    assert not valid and not accepted and missing


def test_removed_claim_cannot_count_as_covered_issue():
    checked = complete_check()
    checked['checks'][0]['supported'] = False
    accepted, missing, _, valid = verification_result(checked, [{'index': 0}], ['rule', 'exceptions'])
    assert valid and not accepted and missing == ['rule', 'exceptions']


@pytest.mark.asyncio
@pytest.mark.parametrize('malformed', [True, False])
async def test_pipeline_repair_and_final_status_follow_verifier_coverage(monkeypatch, malformed):
    import services.evidence_pipeline as pipeline
    evidence = [{'id': 'e1', 'content': 'literal rule', 'kind': 'corpus', 'title': 'source', 'url': ''}]
    async def retrieve(*args):
        return evidence
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)

    class Gateway:
        def __init__(self):
            self.calls = []
        async def json(self, stage, payload):
            self.calls.append(stage)
            if stage == 'understand':
                return {'standalone_question': 'question', 'issues': ['rule', 'exceptions']}
            if stage == 'coverage':
                return {'missing': ['exceptions']}
            if stage in ('answer', 'repair'):
                return {'claims': [{'text': 'rule', 'source_ids': ['e1']}], 'missing': [], 'conflicts': []}
            if stage == 'verify':
                assert payload['initial_coverage']['missing'] == ['exceptions']
                checked = complete_check()
                if malformed:
                    checked.pop('issue_checks')
                return deepcopy(checked)
            raise AssertionError(stage)

    gateway = Gateway()
    trace={}
    result = await run_pipeline('question', [], None, gateway, trace, enable_web=False)
    if malformed:
        assert result['status'] == 'insufficient' and not result['sources']
        assert gateway.calls.count('repair') == 1
    else:
        assert result['status'] == 'partial' and 'מידע חסר' in result['text']
        assert trace['verification_attempts'][0]['verification']['issue_checks'][1]['status']=='missing'
        assert [s['id'] for s in result['sources']] == ['e1']
def test_unverified_gap_and_conflict_prose_cannot_publish_regulatory_claims():
    from services.evidence_pipeline import render
    text,status,sources=render([],['The exempt amount is 999999 NIS in 2050'],['The law allows 88% tax-free withdrawal'])
    assert status=='insufficient' and not sources
    assert '999999' not in text and '2050' not in text and '88%' not in text
    assert 'מידע חסר' in text and 'סתירות שלא הוכרעו' in text
