from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from services.evidence_contract import bind_units, contract_fingerprint, qualified_text, generation_units
from services.evidence_pipeline import resolve_claims, verification_result, render, run_pipeline


def test_d37_original_qualifying_deposit_condition_survives_answer_generation():
    fixture = json.loads((Path(__file__).resolve().parents[1] / 'eval/d37-qualification-regression.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(fixture['quote'].encode()).hexdigest() == fixture['quote_sha256']
    assert 'ריבית ורווחים אחרים שמקורם בהפקדה המוטבת' in fixture['quote']
    source_id = f"D37-V{fixture['original_sha256']}-P43"
    evidence = [{'id': source_id, 'content': fixture['quote'], 'kind': 'corpus', 'title': 'פקודת מס הכנסה', 'page_start': 43}]
    part = lambda text: {'text': text, 'source_ids': [source_id]}
    units, _ = bind_units({'units': [{'rule': part('קיימת אפשרות למשיכה בפטור בתנאי הסעיף'),
        'scope': [part('עובד בקרן השתלמות')],
        'conditions': [part('ריבית ורווחים אחרים שמקורם בהפקדה המוטבת')],
        'exceptions': [part('בפטירת העובד קיימת הוראה לזכאים לקבלת הסכומים')], 'period': None}], 'missing': []}, evidence)
    claims, errors = resolve_claims({'claims': [{'text': 'קיימת אפשרות למשיכה בפטור בתנאי הסעיף', 'unit_ids': ['U1'], 'conditions': []}]}, evidence, units)
    assert not errors
    displayed, _, _ = render(claims, ['current validity unknown'], [])
    assert 'ריבית ורווחים אחרים שמקורם בהפקדה המוטבת' in displayed
    assert 'תקופת תחולה: לא אומתה' in displayed
    checked = verdict(units)
    checked['unit_checks'][0]['complete'] = False
    # The fixture intentionally lacks the full timing rules: when the checker
    # detects that omission, an otherwise positive claim verdict cannot pass.
    assert verification_result(checked, claims, ['withdrawal'], units)[0] == []


EVIDENCE = [{'id':'D1-Vtest-C1', 'content':'Rule for employees only. Interest relief is limited to qualifying deposits. Except event X. Period 2020.',
             'title':'Original', 'kind':'corpus', 'url':''}]


def component(text):
    return {'text':text, 'source_ids':['D1-Vtest-C1']}


def extraction():
    return {'units':[{'rule':component('ניתן למשוך בכפוף לכל התנאים'),
                     'scope':[component('עובדים שכירים בלבד')],
                     'conditions':[component('ההטבה על הרווחים מוגבלת להפקדה המוטבת')],
                     'exceptions':[component('למעט אירוע X')],
                     'period':component('לפי נוסח המקור לשנת 2020')}], 'missing':[]}


def verdict(units):
    return {'checks':[{'index':0,'supported':True,'scope_preserved':True,
                       'period_consistent':True,'qualifications_preserved':True}],
            'unit_checks':[{'unit_id':u['id'],'complete':True} for u in units],
            'component_checks':[{'component_id':c['id'],'supported':True} for u in units for c in u['components']],
            'missing':[], 'conflicts':[], 'issue_checks':[{'issue_index':0,'status':'covered','claim_indices':[0]}]}


def resolved(units, **updates):
    return resolve_claims({'claims':[dict(text='כלל המשיכה', unit_ids=['U1'], **updates)]}, EVIDENCE, units)


def test_answer_cannot_drop_or_replace_extracted_qualifications():
    units, _ = bind_units(extraction(), EVIDENCE)
    claims, errors = resolved(units, components=[], source_ids=['fabricated'], period_known=False)
    assert not errors
    text, status, sources = render(claims, [], [])
    for field in ('scope','conditions','exceptions'):
        assert extraction()['units'][0][field][0]['text'] in text
    assert 'לפי נוסח המקור לשנת 2020' in text
    assert 'fabricated' not in text
    assert sources[0]['content'] == EVIDENCE[0]['content']
    assert len(claims[0]['components']) == 5


def test_required_notice_contents_cannot_be_replaced_by_generic_answer():
    payload=extraction()
    details='ההודעה כוללת שם נמען, מועד הצטרפות ודמי ניהול; הסכמה לשיווק היא רשות.'
    payload['units'][0]['requirements']=[component(details)]
    units,_=bind_units(payload,EVIDENCE)
    claims,errors=resolved(units,requirements=[])
    assert not errors
    text,_,_=render(claims,[],[])
    assert details in text
    checked=verdict(units)
    part=next(c for c in units[0]['components'] if c['kind']=='requirements')
    next(c for c in checked['component_checks'] if c['component_id']==part['id'])['supported']=False
    assert verification_result(checked,claims,['rule'],units)[0]==[]


def test_provider_contract_requires_explicit_requirements_collection():
    from services.source_protocol import extraction_schema
    schema=extraction_schema(['s'])['properties']['units']['items']
    assert 'requirements' in schema['required']


def test_many_required_fields_are_preserved_within_global_contract_budget():
    payload=extraction()
    payload['units'][0]['requirements']=[component(f'required field {i}') for i in range(27)]
    units,_=bind_units(payload,EVIDENCE)
    required=[c for c in units[0]['components'] if c['kind']=='requirements']
    assert len(required)==27
    assert required[-1]['text']=='required field 26'


def test_global_contract_budget_still_rejects_excess_details_without_truncation():
    payload=extraction()
    payload['units'][0]['requirements']=[component(f'field {i}') for i in range(181)]
    with pytest.raises(ValueError,match='component budget'):
        bind_units(payload,EVIDENCE)


@pytest.mark.parametrize('change', [
    lambda p:p['units'][0].pop('conditions'),
    lambda p:p['units'][0].update(exceptions=None),
    lambda p:p['units'][0].pop('period'),
    lambda p:p['units'][0]['conditions'][0].update(source_ids=['made-up']),
    lambda p:p['units'][0]['scope'][0].update(text=''),
])
def test_malformed_extraction_fails_closed(change):
    payload=extraction();change(payload)
    with pytest.raises(ValueError):bind_units(payload,EVIDENCE)


@pytest.mark.parametrize('dimension', ['scope_preserved','period_consistent','qualifications_preserved'])
def test_wrong_population_period_or_dropped_condition_overrules_supported(dimension):
    units,_=bind_units(extraction(),EVIDENCE)
    claims,_=resolved(units)
    checked=verdict(units);checked['checks'][0][dimension]=False
    accepted,missing,_,valid=verification_result(checked,claims,['rule'],units)
    assert valid and not accepted and missing == ['rule']


def test_condition_missed_by_extractor_blocks_even_supported_claim():
    payload=extraction();payload['units'][0]['conditions']=[]
    units,_=bind_units(payload,EVIDENCE)
    claims,_=resolved(units)
    checked=verdict(units);checked['unit_checks'][0]['complete']=False
    assert verification_result(checked,claims,['rule'],units)[0] == []


def test_unsupported_component_blocks_entire_bound_rule():
    units,_=bind_units(extraction(),EVIDENCE)
    claims,_=resolved(units);checked=verdict(units)
    checked['component_checks'][2]['supported']=False
    assert verification_result(checked,claims,['rule'],units)[0] == []


@pytest.mark.parametrize('change', [
    lambda v:v.pop('component_checks'),
    lambda v:v['component_checks'].pop(),
    lambda v:v['component_checks'].append(dict(v['component_checks'][0])),
    lambda v:v['unit_checks'][0].update(complete='true'),
    lambda v:v['checks'][0].pop('period_consistent'),
])
def test_partial_contract_verifier_response_is_not_approval(change):
    units,_=bind_units(extraction(),EVIDENCE)
    claims,_=resolved(units);checked=verdict(units);change(checked)
    accepted,missing,_,valid=verification_result(checked,claims,['rule'],units)
    assert not valid and not accepted and missing


def test_unknown_period_cannot_be_replaced_with_answer_model_year():
    payload=extraction();payload['units'][0]['period']=None
    payload['units'][0]['conditions']=[component('שיעור של 25%')]
    units,_=bind_units(payload,EVIDENCE)
    claims,errors=resolved(units,applicable_year=2026)
    assert not claims and errors == ['0:unscoped_numeric_parameter']


def test_notice_deadline_does_not_establish_legal_period_for_numeric_claim():
    payload=extraction()
    payload['units'][0]['period']={**component('בתוך 14 ימי עסקים'),'start_date':None,'end_date':None}
    payload['units'][0]['conditions']=[component('שיעור של 25%')]
    units,_=bind_units(payload,EVIDENCE)
    assert units[0]['period_known'] is False
    assert any(c['kind']=='temporal_context' and c['text']=='בתוך 14 ימי עסקים' for c in units[0]['components'])
    claims,errors=resolved(units,applicable_year=2026)
    assert not claims and errors==['0:unscoped_numeric_parameter']


def test_calendar_applicability_is_preserved_for_independent_source_verification():
    payload=extraction()
    payload['units'][0]['period']={**component('תחילה ביום 1 בינואר 2020'),'start_date':'2020-01-01','end_date':None}
    units,_=bind_units(payload,EVIDENCE)
    assert units[0]['period_known'] is True
    assert units[0]['applicability_dates']=={'start_date':'2020-01-01'}


@pytest.mark.parametrize('start,end',[('2020-02-31',None),('2020-02-01','2019-01-01'),('14 ימי עסקים',None)])
def test_invalid_or_reversed_calendar_period_is_rejected(start,end):
    payload=extraction()
    payload['units'][0]['period']={**component('period'),'start_date':start,'end_date':end}
    with pytest.raises(ValueError,match='applicability date'):
        bind_units(payload,EVIDENCE)


def test_unknown_unit_cannot_be_replaced_by_valid_citation():
    units,_=bind_units(extraction(),EVIDENCE)
    claims,errors=resolve_claims({'claims':[{'text':'claim','unit_ids':['U999'],
                                          'source_ids':['D1-Vtest-C1']}]},EVIDENCE,units)
    assert not claims and errors


@pytest.mark.asyncio
async def test_repair_cannot_bypass_frozen_contract(monkeypatch):
    import services.evidence_pipeline as pipeline
    async def retrieve(*args):return EVIDENCE
    monkeypatch.setattr(pipeline,'retrieve',retrieve)
    class Gateway:
        def __init__(self):self.stages=[];self.units=None
        async def json(self,stage,payload,**kwargs):
            self.stages.append(stage)
            if stage=='understand':return {'standalone_question':'שאלה','issues':['rule']}
            if stage=='coverage':return {'missing':[]}
            if stage=='evidence_units':return extraction()
            if stage in ('answer','repair'):
                if self.units is None:self.units=deepcopy(payload['units'])
                assert payload['units']==self.units
                return {'claims':[{'text':'כלל ללא פירוט תנאים','unit_ids':['U1'],'conditions':[]}], 'missing':[],'conflicts':[]}
            if stage=='verify':
                assert 'ההטבה על הרווחים מוגבלת להפקדה המוטבת' in payload['claims'][0]['display_text']
                checked=verdict(payload['units'])
                checked['checks'][0]['period_consistent']=False
                return checked
            raise AssertionError(stage)
    gateway=Gateway();trace={}
    result=await run_pipeline('שאלה',[],None,gateway,trace,enable_web=False)
    assert result['status']=='insufficient' and not result['sources']
    assert gateway.stages.count('repair')==1 and gateway.stages.count('evidence_units')==1
    assert generation_units(trace['evidence_units'])==gateway.units
    assert trace['evidence_contract_hash']==contract_fingerprint(bind_units(extraction(),EVIDENCE)[0])
    assert 'כלל ללא פירוט תנאים' not in result['text']
