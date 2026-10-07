import pytest
from services.providers import Gateway


@pytest.mark.asyncio
@pytest.mark.parametrize('budget',[-1,True,4096,30000,'1024'])
async def test_invalid_thinking_budget_fails_before_reserving_or_contacting_provider(budget):
    gateway=Gateway()
    with pytest.raises(ValueError,match='Thinking budget'):
        await gateway.json('audit',{},max_output=4096,thinking_budget=budget)
    assert gateway.spent==0 and gateway.calls==[]
