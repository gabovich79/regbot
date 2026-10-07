from services.evidence_contract import compose_unit_claims, bind_claim, qualified_text


def test_composition_preserves_rule_and_bound_qualifications_without_inventing_year():
    units=[{'id':'U1','period_known':False,'components':[
        {'id':'U1:rule:0','kind':'rule','text':'Original rule','source_ids':['s']},
        {'id':'U1:exceptions:0','kind':'exceptions','text':'Mandatory exception','source_ids':['s']}]}]
    result=compose_unit_claims(units)
    assert result['claims']==[{'text':'Original rule','unit_ids':['U1'],'applicable_year':None}]
    assert 'Mandatory exception' in qualified_text(bind_claim(result['claims'][0], units))
    assert compose_unit_claims(units, True)['claims'][0]['applicable_year'] is None
    assert compose_unit_claims(units, 2020)['claims'][0]['applicable_year']==2020
