import pytest
from scripts.requirement_audit import audit_requirements,validate_verdict


@pytest.mark.parametrize('status,ids,gaps,valid',[
    ('covered',['A1'],[],True),('covered',[],[],False),
    ('covered',['A1'],['Missing population'],False),
    ('partial',['A1'],['Missing population'],True),
    ('partial',['A1'],[],False),('missing',[],['All parts'],True),
    ('missing',['A1'],['All parts'],False),('covered',['A99'],[],False),
    ('conflict',['A1'],[],True),('conflict',[],['Wrong condition'],False),
])
def test_partial_or_unpointed_support_cannot_be_full_credit(status,ids,gaps,valid):
    assert validate_verdict({'status':status,'span_ids':ids,'missing_parts':gaps,'reason':'explanation'},
                            {'A1':'observed'}) is valid


@pytest.mark.asyncio
async def test_routes_do_not_leak_answer_retrieval_or_unrelated_reference():
    case={'question':'q','requirements':[{'id':'r','text':'needed rule','evidence_ids':['ref']}]}
    refs=[{'id':'ref','quote':'expected meaning'},{'id':'other','quote':'irrelevant reference'}]
    seen=[]
    class Gateway:
        async def json(self,stage,payload,**kwargs):
            assert kwargs['thinking_budget']==1024 and kwargs['max_output']>1024
            seen.append(payload)
            assert 'question' not in payload
            assert payload['reference_excerpts']==[{'quote':'expected meaning'}]
            if payload['route']=='answer':
                assert payload['observed_spans']=={'A1':'actual answer'}
            else:
                assert payload['observed_spans']=={'E1':{'source_id':'s','quote':'retrieved only'}}
            return {'status':'missing','span_ids':[],'missing_parts':['needed rule'],'reason':'absent'}
    result=await audit_requirements(case,refs,{'text':'actual answer'},[{'id':'s','content':'retrieved only'}],Gateway())
    assert len(seen)==2 and result['structurally_valid']
    assert not result['claim_safety_evaluated'] and not result['acceptance_passed']


@pytest.mark.asyncio
async def test_missing_reference_fails_before_any_provider_call():
    class Gateway:
        async def json(self,*args,**kwargs):pytest.fail('No paid call with absent reference')
    with pytest.raises(ValueError,match='reference evidence'):
        await audit_requirements({'question':'q','requirements':[{'id':'r','text':'x','evidence_ids':['missing']}]},
                                 [],{'text':''},[],Gateway())
