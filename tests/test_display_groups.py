from copy import deepcopy

from services.evidence_pipeline import display_groups, render


def claim(text):
    source={'id':'s','content':'literal original','title':'source','kind':'corpus','section':'rule','page_start':1}
    return {'text':text,'unit_ids':['U1'],'source_ids':['s'],'period_known':False,'applicable_year':None,
            'components':[{'id':'U1:conditions:0','kind':'conditions','text':'ONLY for population A','source_ids':['s']}],
            'evidence':[source]}


def test_identical_bound_conditions_display_once_without_losing_claims_or_sources():
    claims=[claim('First verified rule'),claim('Second verified rule'),claim('First verified rule')]
    before=deepcopy(claims)
    text,status,sources=render(claims,['unknown validity'],[])
    assert text.count('ONLY for population A')==1
    assert text.count('First verified rule')==1 and text.count('Second verified rule')==1
    assert status=='partial' and sources==[claims[0]['evidence'][0]]
    assert claims==before


def test_distinct_population_or_period_cannot_share_qualifications():
    base=claim('First rule');other=claim('Second rule')
    other['components'][0]['text']='ONLY for population B'
    assert len(display_groups([base,other]))==2
    other=claim('Same text');other['applicable_year']=2020
    assert len(display_groups([base,other]))==2
    other=claim('Same text');other['unit_ids']=['U2']
    assert len(display_groups([base,other]))==2


def test_unbound_claims_are_not_silently_deduplicated():
    a=claim('same');a.pop('components')
    assert len(display_groups([a,deepcopy(a)]))==2
