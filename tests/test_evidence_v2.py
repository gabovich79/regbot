import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
import httpx

from models import database
from models.evidence_store import stage, review, activate, active_chunks
from services.knowledge import prepare_document, ENC, quality_issues
from services.evidence_search import fused_candidates
from services.evidence_pipeline import resolve_claims, run_pipeline
from services.public_access import reserve, settle, identity, assert_owner
from services.web_evidence import source_kind, sanitized_search
from fastapi import HTTPException


@pytest_asyncio.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database,'DB_PATH', str(tmp_path/'test.db'))
    await database.init_db()
    connection = await database.get_db()
    yield connection
    await connection.close()


@pytest.mark.asyncio
async def test_large_parent_read_shares_metadata_and_preserves_review_gate(db):
    from array import array
    from models.evidence_store import version_chunks
    await db.execute("INSERT INTO documents(id,title,source_type,text_path) VALUES(1,'a','pdf','a')")
    await db.commit()
    parent='Original legal section. '*10000
    chunks=[dict(content=f'passage {i}',context='',section='section',section_text=parent) for i in range(20)]
    version=await stage(db,1,'hash','model',{'summary':'summary '*1000},[],chunks,[array('d',[.1,.2]) for _ in chunks])
    assert await version_chunks(db,[version]) == []
    await review(db,version,True)
    loaded=await version_chunks(db,[version])
    assert [c['content'] for c in loaded] == [c['content'] for c in chunks]
    assert all(c['section_text']==parent for c in loaded)
    assert len({id(c['section_text']) for c in loaded})==1
    assert len({id(c['card']) for c in loaded})==1
    assert json.loads(loaded[0]['embedding']) == [.1,.2]


@pytest.mark.asyncio
@pytest.mark.parametrize('provider_fails', [False, True])
async def test_chat_sse_persists_only_checked_answer_and_settles(db, monkeypatch, provider_fails):
    import main
    import services.evidence_pipeline as pipeline
    from config import DEFAULT_MODEL, EMBEDDING_MODEL
    monkeypatch.setenv('DEMO_SESSION_SECRET', 's' * 32)
    monkeypatch.setenv('COOKIE_SECURE', '0')
    monkeypatch.setenv('PROVIDER_PRICES_JSON', json.dumps({
        DEFAULT_MODEL: {'input': 1, 'output': 1},
        EMBEDDING_MODEL: {'input': 1, 'output': 0}}))
    await db.execute("INSERT INTO active_index VALUES(1,'test-release')")
    await db.commit()

    async def checked_answer(question, history, connection, gateway, trace, progress):
        assert not history
        await progress('Checking sources')
        if provider_fails:
            raise RuntimeError('private provider error')
        return {'status': 'supported', 'text': 'Verified final answer',
                'sources': [{'id': 'D1-Vtest-C1', 'content': 'Literal evidence'}]}

    monkeypatch.setattr(pipeline, 'run_pipeline', checked_answer)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
        response = await client.post('/api/chat', data={'question': 'Question'})
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        types = [event['type'] for event in events]
        assert types == (['thinking', 'error', 'done'] if provider_fails else ['thinking', 'sources', 'text', 'usage', 'done'])
        assert 'private provider error' not in response.text
        conversation = events[-1]['conversation_id']
        messages = (await client.get(f'/api/conversations/{conversation}')).json()
        assert [m['role'] for m in messages] == (['user'] if provider_fails else ['user', 'assistant'])
        request_id = response.headers['X-Request-ID']
        reservation = await (await db.execute('SELECT * FROM demo_requests WHERE id=?', (request_id,))).fetchone()
        assert reservation['status'] == ('failed' if provider_fails else 'supported')
        assert reservation['actual'] == 0
        saved_trace = await (await db.execute('SELECT * FROM request_traces WHERE id=?', (request_id,))).fetchone()
        assert saved_trace is not None


def test_preserve_preamble_short_sections_and_cross_page_spans():
    pages = [{'page_number':1,'text':'מבוא חשוב\nסעיף 1\nחל על כל עובד.\nהמשפט מתחיל'},
             {'page_number':2,'text':'ומסתיים כאן.\nסעיף 2\nאסור.\nסעיף 3\nמותר.'}]
    _,card,issues,chunks = prepare_document('',{'id':1,'title':'מקור'},pages)
    assert not any(i in issues for i in ('empty_extraction','invalid_characters'))
    joined = ''.join(c['content'] for c in chunks)
    assert 'מבוא חשוב' in joined and 'אסור.' in joined and 'מותר.' in joined
    section = next(c for c in chunks if c['section'].startswith('סעיף 1'))
    assert section['page_start'] == 1 and section['page_end'] == 2
    assert 'ומסתיים כאן.' in section['content']


def test_chunk_cap_preserves_hebrew_and_short_tail():
    text = 'משפט בעברית. '*3000 + '\nסעיף 2\nלא.'
    _,_,_,chunks = prepare_document(text,{'id':1,'title':'בדיקה'})
    assert all(len(ENC.encode(c['content'])) <= 1200 for c in chunks)
    assert all('\ufffd' not in c['content'] for c in chunks)
    assert chunks[-1]['content'].endswith('לא.')


def test_invalid_text_and_scan_not_ready():
    assert 'invalid_characters' in quality_issues('a\x00b')
    assert quality_issues('', [{'text':''}])


@pytest.mark.asyncio
async def test_staging_missing_original_never_calls_provider(db):
    from services.knowledge import stage_document
    class NoCalls:
        async def json(self,*args,**kwargs):
            pytest.fail('Missing originals must fail before spending')
    with pytest.raises(ValueError,match='missing_original'):
        await stage_document(db,{'id':1,'title':'מקור'},'טקסט מקור ארוך מספיק לצורך בדיקת החילוץ והשמירה.',gateway=NoCalls())


@pytest.mark.asyncio
async def test_staging_card_uses_original_pages_and_shared_gateway(db,tmp_path):
    from services.knowledge import stage_document
    original=tmp_path/'source.pdf';original.write_bytes(b'source bytes used for checksum')
    source='טקסט המקור מהעמוד מופיע כאן במלואו לצורך בדיקת האינדוקס.'
    class FakeGateway:
        async def json(self,stage,payload,**kwargs):
            assert payload['source']==source
            return {'summary':'תקציר','relations':[{'target':'מקור אחר','quote':source}]}
        async def embed(self,texts):
            return [[1.,0.] for _ in texts]
    version,_=await stage_document(db,{'id':1,'title':'מקור','original_path':str(original)},'',
                                   [{'page_number':1,'text':source}],gateway=FakeGateway())
    row=await (await db.execute('SELECT card FROM evidence_versions WHERE id=?',(version,))).fetchone()
    card=json.loads(row['card'])
    assert card['relations'][0]['quote']==source
    assert card['original_checksum']


def test_keyword_match_does_not_remove_semantic_candidates():
    card = json.dumps({'title':'מקור','summary':'זכויות עמית','embedding':[1.,0.]})
    base = {'document_id':1,'card':card,'context':'','embedding':'[1,0]'}
    chunks = [dict(base,id='a',content='סכומים שמשך עובד מחשבונו'),
              dict(base,id='b',content='משיכה בקרן השתלמות דיווח טכני בלבד',embedding='[0,1]')]
    assert {c['id'] for c in fused_candidates('מה כללי משיכה בקרן השתלמות?', [1,0], chunks)} == {'a','b'}


@pytest.mark.asyncio
async def test_stage_review_activation_and_rollback(db):
    await db.execute("INSERT INTO documents(id,title,source_type,text_path) VALUES(1,'a','pdf','a.txt')")
    await db.commit()
    chunk = {'content':'טקסט מקורי','context':'מסמך א','section':'סעיף 1','section_text':'טקסט מקורי'}
    v1 = await stage(db,1,'hash','model',{},[],[chunk],[[1,0]])
    assert await active_chunks(db) == (None,[])
    with pytest.raises(ValueError):
        await activate(db,[v1])
    await review(db,v1,True)
    release = await activate(db,[v1])
    v2 = await stage(db,1,'hash2','model',{},[],[dict(chunk,content='שינוי')],[[0,1]])
    assert (await active_chunks(db))[0] == release
    await review(db,v2,True)
    await activate(db,[v2])
    await activate(db,[v1])
    assert (await active_chunks(db))[1][0]['content'] == 'טקסט מקורי'
    assert (await active_chunks(db))[1][0]['id'].startswith('D1-V')


@pytest.mark.asyncio
async def test_activation_rejects_partial_manifest_and_bad_extraction(db):
    await db.execute("INSERT INTO documents(id,title,source_type,text_path) VALUES(1,'a','pdf','a')")
    await db.commit()
    chunk = dict(content='a',context='b',section='s',section_text='a')
    version = await stage(db,1,'hash','model',{},['corrupt'],[chunk],[[1]])
    with pytest.raises(ValueError):
        await review(db,version,True)
    with pytest.raises(ValueError):
        await activate(db,[])


def test_resolver_rejects_fabrication_and_binds_literal_text():
    e = {'id':'D1-Vabc-C1','content':'שש שנים לפי התנאים שבסעיף.'}
    claims,errors = resolve_claims({'claims':[{'text':'כלל','source_ids':['fake']}]},[e])
    assert errors and not claims
    claims,errors = resolve_claims({'claims':[{'text':'כלל','source_ids':[e['id']]}]},[e])
    assert not errors and claims[0]['evidence'][0]['content'] == e['content']
    assert claims[0]['evidence'][0]['span_hash']
    assert resolve_claims({'claims':[{'text':'תקרה 123 ₪','source_ids':[e['id']]}]},[e])[1]


def test_allowlist_and_private_query():
    assert source_kind('https://www.gov.il/a') == 'official_web'
    assert source_kind('https://www.kolzchut.org.il/a') == 'secondary_web'
    for url in ('http://gov.il','https://gov.il.evil.test','https://127.0.0.1','https://user:pass@gov.il','https://gov.il:8080'):
        with pytest.raises(ValueError):
            source_kind(url)
    query = sanitized_search({'product':'קרן השתלמות','operation':'משיכה','population':'דוד 123456789','question':'סוד'})
    assert 'דוד' not in query and '123456789' not in query and 'סוד' not in query


def test_session_cannot_be_chosen_or_forged(monkeypatch):
    monkeypatch.setenv('DEMO_SESSION_SECRET','s'*32)
    request = SimpleNamespace(cookies={})
    owner,token = identity(request)
    assert identity(SimpleNamespace(cookies={'regbot_session_v2':token}))[0] == owner
    assert identity(SimpleNamespace(cookies={'regbot_session_v2':token+'x'}))[0] != owner


@pytest.mark.asyncio
async def test_ownership_blocks_legacy_and_other_visitors(db):
    await db.execute('INSERT INTO owned_conversations VALUES(1,?)',('alice',))
    await db.commit()
    await assert_owner(db,1,'alice')
    for cid,owner in [(1,'bob'),(2,'alice')]:
        with pytest.raises(HTTPException) as exc:
            await assert_owner(db,cid,owner)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_budget_concurrency_and_fail_closed(db,monkeypatch):
    from config import DEFAULT_MODEL,EMBEDDING_MODEL
    monkeypatch.setenv('PROVIDER_PRICES_JSON',json.dumps({DEFAULT_MODEL:{'input':1,'output':1},EMBEDDING_MODEL:{'input':1,'output':0}}))
    await reserve(db,'r1','alice','ip1')
    with pytest.raises(HTTPException):
        await reserve(db,'r2','alice','ip1')
    await reserve(db,'r3','bob','ip2')
    with pytest.raises(HTTPException):
        await reserve(db,'r4','charlie','ip3')
    await settle(db,'r1',.1,'done')
    await settle(db,'r3',.1,'done')
    for i in range(9):
        await reserve(db,f'cost{i}',f'user{i}',f'ip{i}')
        await settle(db,f'cost{i}',.5,'done')
    with pytest.raises(HTTPException):
        await reserve(db,'over','new','new')
    monkeypatch.setenv('PROVIDER_PRICES_JSON','{}')
    with pytest.raises(HTTPException) as exc:
        await reserve(db,'unpriced','new','new')
    assert exc.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize('history', [[], [{'role':'user','content':'שאלתי על ביטוח אובדן כושר עבודה'}]])
async def test_planner_cannot_replace_first_turn_with_generic_translation(monkeypatch, history):
    import services.evidence_pipeline as pipeline
    question = 'מה התנאים בביטוח אובדן כושר עבודה לפי חוזר 2020-1-22?'
    rewrite = 'What are the conditions for this financial product?'
    captured = {}
    class RetrievalReached(Exception):
        pass
    async def inspect_plan(db, plan, gateway, trace):
        captured.update(plan)
        raise RetrievalReached
    class Fake:
        async def json(self, stage, payload, **kwargs):
            assert stage == 'understand'
            return {'standalone_question':rewrite, 'issues':['conditions'],
                    'retrieval_queries':['תנאי ביטוח אובדן כושר עבודה']}
    monkeypatch.setattr(pipeline, 'retrieve', inspect_plan)
    trace = {}
    with pytest.raises(RetrievalReached):
        await run_pipeline(question, history, None, Fake(), trace, enable_web=False)
    assert captured['standalone_question'] == (rewrite if history else question)
    assert captured['retrieval_queries'] == ['תנאי ביטוח אובדן כושר עבודה']
    assert trace['understanding_raw']['standalone_question'] == rewrite


@pytest.mark.asyncio
async def test_pipeline_removes_unsupported_after_one_repair(monkeypatch):
    import services.evidence_pipeline as pipeline
    evidence = [{'id':'D1-Va-C1','content':'המקור אינו קובע שיעור מס.','title':'מקור','kind':'corpus','url':''}]
    async def fake_retrieve(*args):
        return evidence
    monkeypatch.setattr(pipeline,'retrieve',fake_retrieve)
    class Fake:
        stages = []
        async def json(self,stage,payload,**kwargs):
            self.stages.append(stage)
            if stage=='understand': return {'standalone_question':'שאלה','issues':['שיעור מס']}
            if stage=='coverage': return {'missing':[]}
            if stage=='evidence_units': return {'units':[{'rule':{'text':'כלל','source_ids':['D1-Va-C1']},'scope':[], 'conditions':[], 'exceptions':[], 'period':None}], 'missing':[]}
            if stage in ('answer','repair'): return {'claims':[{'text':'המס 25% בשנת 2026','applicable_year':2026,'unit_ids':['U1']}]}
            if stage=='verify': return {'checks':[{'index':0,'supported':False,'reason':'not in source'}],'missing':['שיעור מס']}
    gateway = Fake()
    result = await run_pipeline('שאלה',[],None,gateway,{},enable_web=False)
    assert '25%' not in result['text']
    assert result['status']=='insufficient'
    assert gateway.stages.count('repair') == 1
    assert not result['sources']


@pytest.mark.asyncio
async def test_http_sessions_isolate_messages(db,monkeypatch):
    import main
    monkeypatch.setenv('DEMO_SESSION_SECRET','s'*32)
    monkeypatch.setenv('COOKIE_SECURE','0')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as first:
        response = await first.get('/api/conversations')
        assert response.status_code == 200 and response.json()==[]
        owner,_ = identity(SimpleNamespace(cookies=first.cookies))
        await db.execute("INSERT INTO conversations(id,session_id) VALUES(100,'old')")
        await db.execute('INSERT INTO owned_conversations VALUES(100,?)',(owner,))
        await db.execute("INSERT INTO messages(conversation_id,role,content) VALUES(100,'user','private')")
        await db.commit()
        assert (await first.get('/api/conversations/100')).status_code == 200
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as other:
            assert (await other.get('/api/conversations')).json()==[]
            assert (await other.get('/api/conversations/100')).status_code == 404
            assert (await other.post('/api/chat',data={'question':'x','conversation_id':100,'session_id':owner})).status_code==404
