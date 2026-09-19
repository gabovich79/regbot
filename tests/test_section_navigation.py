import json
import pytest

from services.section_navigation import catalog,discover_sections,merge_routes


def chunk(n,section,version='v1'):
    return {'id':f'{version}-{n}','version_id':version,'ordinal':n,'section':section,
            'section_text':section+' original text','content':'original chunk',
            'card':json.dumps({'title':version})}


@pytest.mark.asyncio
async def test_rule_outside_direct_shortlist_is_resolved_to_original_chunk():
    rule=chunk(0,'Operative rule');form=chunk(1,'Application form');all_chunks=[rule,form]
    class Gateway:
        async def json(self,stage,payload,**kwargs):
            assert stage=='section_navigation'
            return {'ids':[next(e['id'] for e in payload['sections'] if e['section']=='Operative rule')]}
    trace={}
    found=await discover_sections([form],all_chunks,{'standalone_question':'requirements'},Gateway(),trace)
    assert found==[rule,form] and found[0] is rule
    assert trace['section_navigation']['selected'][0]['source_id']==rule['id']


@pytest.mark.asyncio
async def test_unknown_navigation_pointer_is_not_source_evidence():
    c=chunk(0,'Source')
    class Gateway:
        async def json(self,*args,**kwargs):return {'ids':['invented']}
    with pytest.raises(ValueError,match='navigation pointers'):
        await discover_sections([c],[c],{},Gateway(),{})


def test_noncontiguous_same_heading_keeps_distinct_navigation_targets():
    chunks=[chunk(0,'Same'),chunk(1,'Other'),chunk(2,'Same')]
    entries,lookup,omitted=catalog([chunks[0]],chunks)
    assert len(entries)==3 and not omitted
    assert [g[0]['ordinal'] for g in lookup.values()]==[0,1,2]


def test_navigation_budget_and_route_merge_are_bounded_without_duplicates():
    direct=[chunk(i,f'section {i}') for i in range(50)]
    entries,lookup,omitted=catalog(direct,direct,token_budget=100)
    assert omitted and len(entries)<50
    merged=merge_routes(direct,[direct[49],direct[0]])
    assert len(merged)==40 and len({c['id'] for c in merged})==40
    assert merged[:2]==[direct[49],direct[0]]
