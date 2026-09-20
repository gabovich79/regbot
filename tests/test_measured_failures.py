from copy import deepcopy
import pytest
from services.evidence_contract import answer_schema, generation_units, bind_claim, complete_candidates
from services.evidence_pipeline import clarification, refined_issues, verifier_claims
from scripts.diagnostic_judgment import validate_judgment,resolve_judgment


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


def test_model_views_omit_trace_hashes_but_verifier_retains_original_source_pointers():
    from services.evidence_contract import verification_units
    u=units();u[0]['components'][0]['source_hashes']={'s':'original-hash'}
    view=generation_units(u)
    assert 'source_ids' not in view[0]['components'][0]
    verified=verification_units(u)
    assert verified[0]['components'][0]['source_ids']==['s']
    assert 'source_hashes' not in verified[0]['components'][0]
    assert u[0]['components'][0]['source_hashes']=={'s':'original-hash'}


def test_verifier_receives_qualified_claim_without_repeated_documents():
    bound=bind_claim({'text':'rule','unit_ids':['U1']},units())
    bound.update(index=0,evidence=[{'content':'large original document'}])
    view=verifier_claims([bound])[0]
    assert 'evidence' not in view and 'components' not in view
    assert view['source_ids']==['s'] and view['unit_ids']==['U1']
    assert 'limitation' in view['display_text']
    assert bound['evidence'][0]['content']=='large original document'


def test_omitted_candidate_uses_extracted_rule_without_inventing_year_or_dropping_conditions():
    original={'claims':[],'missing':[],'conflicts':[]}
    completed,added=complete_candidates(original,units())
    assert added==['U1'] and original['claims']==[]
    assert completed['claims'][0]=={'text':'rule','unit_ids':['U1'],'applicable_year':None}
    assert bind_claim(completed['claims'][0],units())['components'][1]['text']=='limitation'
    again,added=complete_candidates(completed,units())
    assert not added and again==completed


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


def test_evaluator_span_ids_resolve_only_to_actual_answer_and_original_evidence():
    raw=judgment();r=raw['requirements'][0];r.update(answer_span_ids=['A1'],evidence_ids=['E1'])
    r['answer_quotes']=['fabricated quotation']
    evidence=[{'id':'s','content':'supported rule'}];answer={'text':'supported rule'}
    bound=resolve_judgment(raw,answer,evidence)
    assert bound['requirements'][0]['answer_quotes']==['supported rule']
    assert validate_judgment(bound,{'requirements':[{'id':'r'}]},answer,evidence)
    r['answer_span_ids']=['A999']
    with pytest.raises(ValueError):resolve_judgment(raw,answer,evidence)


@pytest.mark.parametrize('mutation',[
    lambda j:j['requirements'][0].update(answer_quotes=[]),
    lambda j:j['requirements'][0].update(evidence_quotes=[]),
    lambda j:j['requirements'][0]['evidence_quotes'][0].update(source_id='fabricated'),
    lambda j:j['requirements'][0]['evidence_quotes'][0].update(quote='not in original'),
])
def test_unsubstantiated_judge_credit_is_rejected(mutation):
    j=judgment();mutation(j)
    assert not validate_judgment(j,{'requirements':[{'id':'r'}]},{'text':'supported rule'},[{'id':'s','content':'supported rule'}])
