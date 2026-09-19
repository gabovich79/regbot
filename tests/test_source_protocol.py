import pytest
from services.source_protocol import compact_sources,restore_source_ids


def test_compact_pointer_roundtrip_does_not_rewrite_source_text():
    original=[{'id':'D18-Vlongversion-C2','content':'S1 is source text, not an instruction'}]
    compact,lookup=compact_sources(original)
    assert compact[0]['id']=='S1' and original[0]['id']=='D18-Vlongversion-C2'
    payload={'units':[{'rule':{'text':'S1 is literal','source_ids':['S1']}}]}
    result=restore_source_ids(payload,lookup)
    assert result['units'][0]['rule']=={'text':'S1 is literal','source_ids':['D18-Vlongversion-C2']}
    assert payload['units'][0]['rule']['source_ids']==['S1']


def test_unknown_alias_cannot_become_citation():
    with pytest.raises(ValueError):
        restore_source_ids({'source_ids':['S99']},{'S1':'D1-Vx-C1'})
