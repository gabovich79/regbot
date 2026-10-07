"""Optional persistent, cross-process ceiling for a development campaign.

Reservations survive interrupted calls and process restarts. Keep the ledger
outside replaceable corpus copies; do not use it for the public daily quota.
"""
import math
import os
import sqlite3
import uuid


class CampaignLimit(RuntimeError):
    pass


class Reservation(float):
    def __new__(cls, amount, path=None, ticket=None):
        value = super().__new__(cls, amount)
        value.path, value.ticket = path, ticket
        return value


def reserve(amount, purpose, model):
    path = os.getenv('CAMPAIGN_BUDGET_DB')
    configured = os.getenv('CAMPAIGN_BUDGET_USD')
    if not path and not configured:
        return Reservation(amount)
    if not path or not configured:
        raise CampaignLimit('Campaign ledger and ceiling must both be configured')
    ceiling = float(configured)
    if not math.isfinite(ceiling) or ceiling <= 0 or not math.isfinite(amount) or amount < 0:
        raise CampaignLimit('Invalid campaign budget')
    with sqlite3.connect(path, timeout=10) as db:
        db.execute('CREATE TABLE IF NOT EXISTS campaign_limit(id INTEGER PRIMARY KEY CHECK(id=1), ceiling REAL NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS campaign_calls(ticket TEXT PRIMARY KEY, purpose TEXT, model TEXT, cost REAL NOT NULL, settled INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
        db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT OR IGNORE INTO campaign_limit VALUES(1,?)', (ceiling,))
        stored = db.execute('SELECT ceiling FROM campaign_limit WHERE id=1').fetchone()[0]
        # A different environment must not silently raise an existing cap.
        ceiling = min(ceiling, stored)
        spent = db.execute('SELECT COALESCE(SUM(cost),0) FROM campaign_calls').fetchone()[0]
        if spent + amount > ceiling:
            raise CampaignLimit('Campaign spending ceiling reached; no provider call sent')
        ticket = uuid.uuid4().hex
        db.execute('INSERT INTO campaign_calls(ticket,purpose,model,cost) VALUES(?,?,?,?)', (ticket,purpose,model,amount))
    return Reservation(amount,path,ticket)


def settle(reservation, actual):
    if not getattr(reservation, 'ticket', None):
        return
    if not math.isfinite(actual) or actual < 0:
        raise CampaignLimit('Invalid provider usage; reservation retained')
    with sqlite3.connect(reservation.path, timeout=10) as db:
        cursor = db.execute('UPDATE campaign_calls SET cost=?, settled=1 WHERE ticket=? AND settled=0', (actual,reservation.ticket))
        if cursor.rowcount != 1:
            raise CampaignLimit('Unknown or already settled campaign reservation')
