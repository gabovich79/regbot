from copy import deepcopy
import pytest
from services.evidence_contract import answer_schema, generation_units, bind_claim
from services.evidence_pipeline import clarification, refined_issues
from scripts.diagnostic_judgment import validate_judgment


def units():
    return [{'id':'U1','period_known':False,'components':[
        {'id':'U1:rule:0','kind':'rule','text':'rule','source_ids':['s']},
        {'id':'U1:conditions:0','kind':'conditions','text':'limitation','source_ids':['s']}]}]


def test_generation_can_select_complete_units_only():
    u=units()
    assert answer_schema(u)['properties']['claims']['items']['properties']['unit_ids']['items']['enum']==['U1']
    view=generation_units(u)
    assert all('id' not in c for c in view[0]['components'])
    assert view[0]['components'][1]['text']=='limitation'
    with pytest.raises(ValueError):bind_claim({'unit_ids':['U1:rule:0']},u)
    assert len(bind_claim({'unit_ids':['U1']},u)['components'])==2


def test_only_source_bound_aspects_expand_coverage():
    aspects=[{'issue':'notice deadline','source_ids':['s']},{'issue':'invented','source_ids':['unknown']}]
    assert refined_issues(['notices'],{'aspects':aspects},[{'id':'s'}])==['notices','notice deadline']


def test_personal_clarification_cannot_publish_arbitrary_model_prose():
    result=clarification({'personal_determination':True,'missing_user_facts':['product','year','You qualify for a benefit']})
    assert result['needs_clarification'] and not result['sources']
    assert 'You qualify' not in result['text']
    assert clarification({'personal_determination':False,'missing_user_facts':['product']}) is None


def judgment():
    return {'requirements':[{'id':'r','answered':True,'retrieved':True,'answer_quotes':['supported rule'],
            'evidence_quotes':[{'source_id':'s','quote':'supported rule'}]}],
            'unsupported_claims':[],'incorrect_claims':[],'conflicts':[]}


def test_judge_cannot_credit_retrieved_text_as_answer():
    case={'requirements':[{'id':'r'}]}; evidence=[{'id':'s','content':'supported rule'}]
    assert validate_judgment(judgment(),case,{'text':'supported rule'},evidence)
    assert not validate_judgment(judgment(),case,{'text':'לא נמצאו ראיות מספיקות לתשובה מבוססת לשאלה זו.'},evidence)
    assert not validate_judgment(judgment(),case,{'text':'no answer\n\nמקורות ששימשו בתשובה:\nsupported rule'},evidence)


@pytest.mark.parametrize('mutation',[
    lambda j:j['requirements'][0].update(answer_quotes=[]),
    lambda j:j['requirements'][0].update(evidence_quotes=[]),
    lambda j:j['requirements'][0]['evidence_quotes'][0].update(source_id='fabricated'),
    lambda j:j['requirements'][0]['evidence_quotes'][0].update(quote='not in original'),
])
def test_unsubstantiated_judge_credit_is_rejected(mutation):
    j=judgment();mutation(j)
    assert not validate_judgment(j,{'requirements':[{'id':'r'}]},{'text':'supported rule'},[{'id':'s','content':'supported rule'}])
