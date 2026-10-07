"""Server-owned anonymous sessions and SQLite-atomic public quotas."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException

COOKIE = 'regbot_session_v2'
RESERVATION = 0.50


def secret():
    value = os.getenv('DEMO_SESSION_SECRET','')
    if len(value) < 32:
        raise HTTPException(503, 'Public sessions are not configured')
    return value.encode()


def identity(request):
    key = secret()
    value = request.cookies.get(COOKIE, '')
    parts = value.split('.')
    if len(parts) == 2 and len(parts[0]) == 64 and hmac.compare_digest(parts[1], hmac.new(key, parts[0].encode(), hashlib.sha256).hexdigest()):
        token = parts[0]
    else:
        token = secrets.token_hex(32)
    signature = hmac.new(key, token.encode(), hashlib.sha256).hexdigest()
    owner = hashlib.sha256(token.encode()).hexdigest()
    return owner, token + '.' + signature


def set_cookie(response, value):
    response.set_cookie(COOKIE, value, httponly=True, secure=os.getenv('COOKIE_SECURE','1') == '1',
                        samesite='strict', max_age=30*24*3600)
    response.headers['Cache-Control'] = 'no-store'


def client_ip(request):
    # Never trust arbitrary X-Forwarded-For. A deployment must explicitly set
    # its trusted proxy addresses before accepting forwarded visitor addresses.
    address = request.client.host if request.client else 'unknown'
    trusted = {x.strip() for x in os.getenv('TRUSTED_PROXY_IPS','').split(',') if x.strip()}
    if address in trusted:
        chain = [x.strip() for x in request.headers.get('x-forwarded-for','').split(',') if x.strip()]
        for value in reversed(chain):
            if value not in trusted:
                import ipaddress
                try:
                    address = str(ipaddress.ip_address(value))
                except ValueError:
                    raise HTTPException(400,'Invalid forwarded address')
                break
    return hmac.new(secret(), address.encode(), hashlib.sha256).hexdigest()


def same_origin(request):
    origin = request.headers.get('origin')
    expected = os.getenv('PUBLIC_BASE_URL') or str(request.base_url)
    if origin and origin.rstrip('/') != expected.rstrip('/'):
        raise HTTPException(403,'Cross-origin requests are not allowed')
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403,'Cross-site requests are not allowed')


async def assert_owner(db, conversation_id, owner):
    row = await (await db.execute('SELECT 1 FROM owned_conversations WHERE conversation_id=? AND owner=?', (conversation_id,owner))).fetchone()
    if not row:
        raise HTTPException(404,'Conversation not found')


async def reserve(db, request_id, owner, ip_hash):
    from config import DEFAULT_MODEL, EMBEDDING_MODEL
    from services.providers import rate, BudgetExceeded
    try:
        rate(DEFAULT_MODEL)
        rate(EMBEDDING_MODEL)
    except BudgetExceeded as exc:
        raise HTTPException(503, str(exc))
    now = datetime.now(timezone.utc)
    day = now.astimezone(ZoneInfo('Asia/Jerusalem')).date().isoformat()
    try:
        await db.execute('BEGIN IMMEDIATE')
        # On crashes an unresolved reservation remains charged for the day;
        # only its concurrency lease expires.
        await db.execute("UPDATE demo_requests SET status='abandoned' WHERE status='running' AND expires_at < ?", (now.isoformat(),))
        rows = await (await db.execute('SELECT * FROM demo_requests WHERE day=?', (day,))).fetchall()
        if sum(r['owner'] == owner for r in rows) >= 10 or sum(r['ip_hash'] == ip_hash for r in rows) >= 30:
            raise HTTPException(429,'מכסת ההתנסות היומית הסתיימה.')
        running = await (await db.execute("SELECT owner FROM demo_requests WHERE status='running'")).fetchall()
        if any(r['owner'] == owner for r in running) or len(running) >= 2:
            raise HTTPException(429,'בקשה אחרת עדיין מעובדת. נסה שוב בעוד רגע.')
        spent = sum(r['actual'] if r['actual'] is not None else r['reserved'] for r in rows)
        if spent + RESERVATION > 5.0:
            raise HTTPException(429,'תקציב ההתנסות היומי הסתיים. השירות יחודש ביום הבא.')
        await db.execute('INSERT INTO demo_requests(id,day,owner,ip_hash,reserved,status,expires_at) VALUES(?,?,?,?,?,?,?)',
            (request_id,day,owner,ip_hash,RESERVATION,'running',(now+timedelta(seconds=120)).isoformat()))
        await db.commit()
    except BaseException:
        await db.rollback()
        raise


async def settle(db, request_id, actual, status):
    await db.execute('UPDATE demo_requests SET actual=?,status=? WHERE id=?', (actual,status,request_id))
    await db.commit()
