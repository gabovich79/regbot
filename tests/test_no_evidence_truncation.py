from copy import deepcopy

import pytest

from services.evidence_contract import bind_units, complete_candidates
from services.evidence_pipeline import refined_issues, resolve_claims


def fixture(count):
    evidence=[{'id':'s','content':'Source text'}]
    part={'text':'A source-bound rule','source_ids':['s']}
    unit={'rule':part,'scope':[],'conditions':[],'exceptions':[],'period':None}
    return {'units':[deepcopy(unit) for _ in range(count)],'missing':[]},evidence


def test_late_transition_and_its_condition_survive_binding_and_resolution():
    payload,evidence=fixture(23)
    payload['units'][-1]['rule']['text']='A transitional provision'
    payload['units'][-1]['conditions']=[{'text':'Only earlier agreements','source_ids':['s']}]
    units,missing=bind_units(payload,evidence)
    # Even a full generator response cannot crowd out the late source rule.
    answer={'claims':[{'text':'A rule','unit_ids':['U1'],'applicable_year':None} for _ in range(30)]}
    answer,added=complete_candidates(answer,units)
    resolved,errors=resolve_claims(answer,evidence,units)
    assert not missing and not errors
    assert 'U23' in added
    last=next(c for c in resolved if c['unit_ids']==['U23'])
    assert last['text']=='A transitional provision'
    assert last['components'][1]['text']=='Only earlier agreements'
    assert {u for c in resolved for u in c['unit_ids']}=={u['id'] for u in units}


def test_invalid_late_source_is_checked_instead_of_discarded():
    payload,evidence=fixture(23)
    payload['units'][-1]['rule']['source_ids']=['invented']
    with pytest.raises(ValueError,match='Unknown evidence component source'):
        bind_units(payload,evidence)


def test_total_component_budget_still_blocks_oversized_contract():
    payload,evidence=fixture(100)
    for u in payload['units']:
        u['conditions']=[{'text':'A condition','source_ids':['s']}]
    with pytest.raises(ValueError,match='component budget'):
        bind_units(payload,evidence)


def test_source_discovered_scope_survives_full_question_plan():
    issues=[f'Heading {i}' for i in range(20)]
    result=refined_issues(issues,{'aspects':[{'issue':'A scope exclusion','source_ids':['s']}]},[{'id':'s'}])
    assert len(result)==21 and result[-1]=='A scope exclusion'
