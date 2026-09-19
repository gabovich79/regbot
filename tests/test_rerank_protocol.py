import pytest
from services.rerank_protocol import ranked_ids


def rating(i,score):return {'id':i,'score':score,'reason':'Supports a required aspect'}


def test_late_essential_section_outranks_early_background_independent_of_reply_order():
    lookup={f'S{i}':{} for i in range(30)}
    rows=[rating(i,1) for i in lookup]
    rows[-1]['score']=3
    rows[-2]['score']=0
    result=ranked_ids({'ratings':list(reversed(rows))},lookup)
    assert result[0]=='S29' and 'S28' not in result and len(result)==20
    assert result[1]=='S0'


@pytest.mark.parametrize('rows',[
    [],[rating('S1',3),rating('S1',2)],
    [rating('S1',3),rating('invented',2)],
    [rating('S1',True),rating('S2',2)],
    [rating('S1',4),rating('S2',2)],
])
def test_incomplete_or_invalid_ratings_cannot_silently_select_sources(rows):
    with pytest.raises(ValueError):ranked_ids({'ratings':rows},{'S1':{},'S2':{}})


def test_irrelevant_sources_do_not_fill_a_quota():
    assert ranked_ids({'ratings':[rating('S1',0)]},{'S1':{}})==[]
