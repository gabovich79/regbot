import asyncio
from types import SimpleNamespace

import pytest
from google import genai

from services.providers import Gateway


@pytest.mark.asyncio
@pytest.mark.parametrize('failure',[TimeoutError,asyncio.CancelledError])
async def test_failed_call_keeps_reservation_and_reports_stage_without_sensitive_error(monkeypatch,failure):
    calls=[]
    class Client:
        def __init__(self,**kwargs): self.aio=self; self.models=self
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def generate_content(self,**kwargs):
            calls.append(kwargs)
            raise failure('sensitive provider text must not enter trace')
        def close(self): pass
    monkeypatch.setattr(genai,'Client',Client)
    gateway=Gateway(spent=.07)
    monkeypatch.setattr(gateway,'authorize',lambda *args:(.07,{'input':.3,'output':2.5}))
    with pytest.raises(failure):
        await gateway.json('evidence_units',{'source':'מקור'},max_output=12000)
    assert len(calls)==1 and len(gateway.calls)==1 and gateway.spent==.07
    trace=gateway.calls[0]
    assert trace['stage']=='evidence_units' and trace['cost_status']=='reservation_retained'
    assert trace['cost']==.07 and trace['seconds']>=0 and trace['input_tokens'] is None
    assert trace['error_type']==failure.__name__ and 'sensitive' not in str(trace)
