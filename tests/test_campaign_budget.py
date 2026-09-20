import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from services.spend_guard import reserve, settle, CampaignLimit


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / 'spending.db'
    monkeypatch.setenv('CAMPAIGN_BUDGET_DB', str(path))
    monkeypatch.setenv('CAMPAIGN_BUDGET_USD', '20')
    return path


def test_reservations_survive_restarts_and_cannot_raise_cap(ledger, monkeypatch):
    first = reserve(12, 'indexing', 'model')
    reserve(7, 'evaluation', 'model')
    monkeypatch.setenv('CAMPAIGN_BUDGET_USD', '200')
    with pytest.raises(CampaignLimit):
        reserve(2, 'new_process', 'model')
    settle(first, 1)
    reserve(12, 'evaluation', 'model')
    with pytest.raises(CampaignLimit):
        reserve(.01, 'evaluation', 'model')
    with pytest.raises(CampaignLimit):
        settle(first, 0)


def test_concurrent_reservations_are_atomic(ledger):
    reserve(0, 'initialize', 'model')
    def attempt(_):
        try:
            reserve(3, 'parallel', 'model')
            return True
        except CampaignLimit:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(12))) == 6
    with sqlite3.connect(ledger) as db:
        assert db.execute('SELECT SUM(cost) FROM campaign_calls').fetchone()[0] == 18
