from copy import deepcopy

import pytest

from services.compact_unit_protocol import expand
from services.evidence_contract import bind_units


def payload():
    return {'parts':[
        {'id':'P1','text':'כלל ראשון','source_ids':['S1']},
        {'id':'P2','text':'כלל שני','source_ids':['S1']},
        {'id':'P3','text':'רק אם מתקיים תנאי א או תנאי ב','source_ids':['S2']},
        {'id':'P4','text':'פרט רשות','source_ids':['S1']}],
        'units':[{'rule':'P1','scope':[],'conditions':['P3'],'exceptions':[], 'requirements':['P4'],'period':None},
                 {'rule':'P2','scope':[],'conditions':['P3'],'exceptions':[], 'requirements':[],'period':None}],
        'missing':['חסר מקור לתאריך תחילה']}


def test_shared_parts_expand_to_full_independent_units_with_same_source_bindings():
    value=payload(); original=deepcopy(value); result=expand(value)
    evidence=[{'id':'S1','content':'מקור הכללים'},{'id':'S2','content':'מקור התנאי'}]
    bound,gaps=bind_units(result,evidence)
    assert value==original and len(bound)==2 and gaps==value['missing']
    assert result['units'][0]['conditions']==result['units'][1]['conditions']
    result['units'][0]['conditions'][0]['text']='modified'
    assert result['units'][1]['conditions'][0]['text']=='רק אם מתקיים תנאי א או תנאי ב'
    assert result['units'][1]['requirements']==[]


@pytest.mark.parametrize('mutation',['dangling','duplicate','orphan','missing_list','missing_period'])
def test_invalid_contract_is_not_silently_repaired(mutation):
    value=payload()
    if mutation=='dangling':value['units'][0]['conditions']=['P99']
    elif mutation=='duplicate':value['parts'].append(value['parts'][0])
    elif mutation=='orphan':value['parts'].append({'id':'P99','text':'lost detail','source_ids':['S1']})
    elif mutation=='missing_list':del value['units'][0]['exceptions']
    else:del value['units'][0]['period']
    with pytest.raises(ValueError):expand(value)


def test_unknown_source_still_rejected_by_existing_binder():
    value=payload();value['parts'][2]['source_ids']=['UNKNOWN']
    with pytest.raises(ValueError,match='source'):
        bind_units(expand(value),[{'id':'S1','content':'source'}])
