"""Bounded development diagnostic. Never activates or approves an index.

Runs on a separate copy with a persistent campaign ledger. Reference answers
are prepared before system answers and remain machine-checked, not legal gold.
"""
import argparse,asyncio,hashlib,json,os,sqlite3,sys,time,urllib.request,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
IDS=[1,3,19,22,25,26,34,36,37,38]
CASES=[
 ('withdrawal','מה כללי משיכה בקרן השתלמות?',37,['(16א)','(16ב)','שש שנים','שלוש שנים']),
 ('fees','לפי חוזר 2017-9-15, מה תקופת ההנחה בדמי ניהול ובאילו תנאים אפשר להעלות אותם?',3,[]),
 ('fees-old','מה נקבע בחוזר דמי ניהול 2016-9-8 לגבי העלאת דמי ניהול לאחר הנחה?',19,[]),
 ('join','אילו פרטים והודעות נדרשים בהצטרפות לקרן פנסיה או לקופת גמל לפי חוזר 2021-9-5?',1,['הצטרפות','הודעה']),
 ('transfer','כיצד מטופלת בקשת העברת כספים בין קופות גמל לפי חוזר 2016-9-11, לרבות ביטול הבקשה?',22,['ביטול','בקשת העברה']),
 ('employer','מהן חובות הדיווח וההיזון החוזר בהפקדות מעסיק לפי חוזר 2019-9-14?',25,['היזון','דיווח']),
 ('identity','אילו בדיקות נדרשות לפני ביצוע פעולה על ידי גוף מוסדי לפי חוזר 2015-9-30?',26,[]),
 ('annual-cost','מה כוללת העלות השנתית הצפויה המוצגת לעמית?',34,[]),
 ('loan','לפי חוזר 2016-9-17, מה התנאים והמגבלות למתן הלוואה לעמית מקרן השתלמות?',36,['הלוואות לעמיתים','הלוואה לעמית','80%','50%']),
 ('missing-personal','האם אני זכאי למשיכת כספים ללא מס?',37,['שש שנים','שלוש שנים'])]

def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)

def setup(dest):
    if dest.exists():raise ValueError('New isolated destination required')
    dest.mkdir(parents=True)
    src=sqlite3.connect('file:/var/data/regbot.db?mode=ro',uri=True)
    db=sqlite3.connect(dest/'regbot.db');src.backup(db);src.close()
    db.row_factory=sqlite3.Row
    originals=dest/'originals';originals.mkdir()
    manifest=json.loads((ROOT/'eval/pilot-source-manifest.json').read_text(encoding='utf-8'))
    by_id={x['id']:x for x in manifest}
    receipt=[]
    for doc_id in IDS:
        item=by_id[doc_id];target=originals/(str(doc_id)+item['suffix'])
        if doc_id in [36,37,38]:
            shutil.copyfile('/var/data/originals/'+str(doc_id)+'.pdf',target)
        elif doc_id==1:
            with zipfile.ZipFile('/var/data/backups/RegBot-recovered-originals-2026-09-19.zip') as z:target.write_bytes(z.read('1.pdf'))
        else:
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self,*a,**kw):return None
            opener=urllib.request.build_opener(NoRedirect())
            with opener.open(urllib.request.Request(item['url'],headers={'User-Agent':'RegBot source recovery check'}),timeout=25) as response:
                data=response.read(12*1024*1024+1)
            if len(data)>12*1024*1024:raise ValueError('Source too large')
            target.write_bytes(data)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Source checksum changed')
        db.execute('UPDATE documents SET original_path=?,title=?,source_ref=? WHERE id=?',(str(target),item['title'],item['url'],doc_id))
        receipt.append(item)
    db.commit();db.close();save(dest/'source-receipt.json',receipt)

def extract(document):
    from services.document_service import extract_pdf_pages,extract_docx
    path=Path(document['original_path'])
    pages=extract_pdf_pages(str(path)) if path.suffix=='.pdf' else None
    if pages and document['id']==37:
        # These exact source-hash pages were visually confirmed blank locally.
        for p in pages:
            if p['page_number'] in [273,276,278,281,283,286,288,291] and not p['text'].strip():p['blank_page_confirmed']=True
    text='\n\n'.join(p['text'] for p in pages) if pages else extract_docx(str(path))
    return text,pages

async def execute(args):
    dest=Path(args.data_dir).resolve()
    if dest==Path('/var/data') or not (dest/'source-receipt.json').exists():raise ValueError('Isolated source receipt required')
    os.environ['DATA_DIR']=str(dest)
    os.environ['PROVIDER_PRICES_JSON']=(ROOT/'deployment/provider-prices-2026-09-19.json').read_text()
    if not os.getenv('CAMPAIGN_BUDGET_DB') or os.getenv('CAMPAIGN_BUDGET_USD')!='20':raise ValueError('Persistent $20 campaign cap required')
    from models.database import init_db,get_db
    from services.providers import Gateway,BudgetExceeded
    from services.knowledge import stage_document
    await init_db();db=await get_db()
    gateway=Gateway(purpose='pilot_'+args.phase,limit=args.budget)
    report=[]
    try:
        if args.phase=='stage':
            for doc_id in IDS:
                existing=await (await db.execute('SELECT id FROM evidence_versions WHERE document_id=?',(doc_id,))).fetchone()
                if existing:report.append({'id':doc_id,'version':existing['id'],'status':'already_staged'});continue
                row=dict(await (await db.execute('SELECT * FROM documents WHERE id=?',(doc_id,))).fetchone())
                text,pages=extract(row)
                version,count=await stage_document(db,row,text,pages,gateway=gateway)
                report.append({'id':doc_id,'version':version,'chunks':count,'status':'staged'})
                save(dest/'stage-report.json',{'documents':report,'cost':gateway.spent,'activated':False})
                print(json.dumps(report[-1]),flush=True)
        elif args.phase=='references':
            if (dest/'references.json').exists():raise ValueError('References already frozen')
            if (dest/'references-progress.json').exists():report=[r for r in json.loads((dest/'references-progress.json').read_text()) if r['literal_quotes_valid']]
            for key,question,doc_id,needles in CASES:
                if any(c['id']==key for c in report):continue
                row=dict(await (await db.execute('SELECT * FROM documents WHERE id=?',(doc_id,))).fetchone())
                text,pages=extract(row)
                excerpts=[]
                # Reference annotation only, never supplied to retrieval.
                # Inspected source pages contain section 9(16a/b), including
                # the self-employed paragraph continuing across the page break.
                if doc_id==37:excerpts=['\n\n'.join(p['text'] for p in pages if p['page_number'] in [43,44])]
                elif len(text)<24000:excerpts=[text]
                else:
                    for needle in needles:
                        start=0
                        for _ in range(6):
                            index=text.find(needle,start)
                            if index<0:break
                            excerpts.append(text[max(0,index-500):index+4000]);start=index+len(needle)
                if not excerpts:raise ValueError('No reference excerpts for '+key)
                # Models select immutable source pointers, not rewritten quotes.
                passages={}
                for excerpt in dict.fromkeys(excerpts):
                    for offset in range(0,len(excerpt),1600):
                        passages[f'R{doc_id}-{len(passages)}']=excerpt[offset:offset+1800]
                result=await gateway.json('pilot_reference',{'task':'Prepare independent Hebrew expected answer from original excerpts. Return required_claims [up to 8 short claims], exceptions [up to 4 short exceptions], supporting_ids [IDs of up to 10 supporting excerpts], missing [up to 4 short items]. Maximum 700 words TOTAL. Do NOT transcribe quotes: use the provided IDs. Do not use knowledge absent from source. Broad personal eligibility cannot be determined without user facts. Distinguish historical circular rules from current rules.','question':question,'source_title':row['title'],'excerpts':passages},max_output=8192)
                selected=result.get('supporting_ids',[])
                valid=isinstance(selected,list) and bool(selected) and all(isinstance(s,str) and s in passages for s in selected)
                result['quotes']=[passages[s] for s in dict.fromkeys(selected)] if valid else []
                valid=valid and all(q in text for q in result['quotes'])
                check=await gateway.json('pilot_reference_check',{'task':'Independently assess whether reference claims are supported and sufficiently complete for the question. Return accepted boolean and issues [up to 5 brief items]. Maximum 250 words. This is diagnostic, not professional approval.','question':question,'reference':result,'excerpts':excerpts})
                report.append({'id':key,'question':question,'source_document':doc_id,'source_sha256':hashlib.sha256(Path(row['original_path']).read_bytes()).hexdigest(),'reference':result,'literal_quotes_valid':valid,'independent_check':check,'professional_approved':False})
                save(dest/'references-progress.json',report);print(json.dumps({'reference':key,'quotes_valid':valid,'check':check.get('accepted')}),flush=True)
            save(dest/'references.json',report)
        elif args.phase in ('run','baseline'):
            from services import evidence_search
            from services.evidence_pipeline import run_pipeline
            # Diagnostic-only reader of staged chunks. Never changes review or
            # activation state, and cannot affect the public app process.
            async def diagnostic_chunks(connection):
                from models.evidence_store import version_chunks
                versions=await (await connection.execute('SELECT id FROM evidence_versions ORDER BY document_id')).fetchall()
                return 'UNAPPROVED-DEVELOPMENT-PILOT',await version_chunks(connection,[r['id'] for r in versions],approved_only=False)
            evidence_search.active_chunks=diagnostic_chunks
            references=json.loads((dest/'references.json').read_text())
            if args.phase=='baseline':
                from types import SimpleNamespace
                from services import rag_service,claude_service
                from config import DEFAULT_MODEL,MAX_OUTPUT_TOKENS,GOOGLE_API_KEY
                from google import genai
                from google.genai import types
                # Meter legacy embeddings and generation without changing its
                # retrieval, prompts, generation configuration or postprocessing.
                async def legacy_embeddings(*,model,input,**kwargs):
                    vectors=await gateway.embed([input] if isinstance(input,str) else input)
                    return SimpleNamespace(data=[SimpleNamespace(embedding=v,index=i) for i,v in enumerate(vectors)])
                rag_service.openai_client=SimpleNamespace(embeddings=SimpleNamespace(create=legacy_embeddings))
                def legacy_generate(instructions,history,message):
                    reserved,pricing=gateway.authorize(DEFAULT_MODEL,instructions+str(history)+message,MAX_OUTPUT_TOKENS)
                    config=claude_service.build_generation_config(instructions,enable_google_search=False)
                    with genai.Client(api_key=GOOGLE_API_KEY,http_options=types.HttpOptions(timeout=85000,retry_options=types.HttpRetryOptions(attempts=1))) as client:
                        response=client.models.generate_content(model=DEFAULT_MODEL,contents=list(history)+[types.Content(role='user',parts=[types.Part(text=message)])],config=config)
                    usage=response.usage_metadata
                    if usage is None:raise ValueError('Missing baseline provider usage')
                    inputs=usage.prompt_token_count or 0;outputs=(usage.candidates_token_count or 0)+(usage.thoughts_token_count or 0)
                    actual=(inputs*pricing['input']+outputs*pricing['output'])/1000000
                    asyncio.run(gateway.record(DEFAULT_MODEL,reserved,actual,inputs,outputs,'baseline_answer'))
                    return [response.text],{'input_tokens':inputs,'output_tokens':outputs}
                claude_service._sync_send_and_collect=legacy_generate
            for case in references:
                trace={};start=time.monotonic();before=gateway.spent
                try:
                    if args.phase=='run':
                        answer=await asyncio.wait_for(run_pipeline(case['question'],[],db,gateway,trace,enable_web=False),timeout=90)
                    else:
                        events=[]
                        async def baseline():
                            async for event in claude_service.stream_chat(case['question'],db):events.append(event)
                        await asyncio.wait_for(baseline(),timeout=90)
                        answer={'text':''.join(e.get('text','') for e in events if e['type']=='text'),'errors':[e for e in events if e['type']=='error']}
                        trace['events']=events
                    answer_seconds=time.monotonic()-start
                    if case['literal_quotes_valid'] and case['independent_check'].get('accepted') is True:
                        judgment=await gateway.json('pilot_judge',{'task':'Compare answer against independent reference and supplied original quotes. Return correct boolean, complete boolean, unsupported_claims [text], missing_claims [text], notes [text]. Refusing to answer safely is not a complete answer. Do not treat an automatic supported label as proof.','case':case,'answer':answer})
                    else:
                        judgment={'review_required':True,'reason':'Reference failed literal or independent validation; no accuracy score permitted'}
                    result={'id':case['id'],'question':case['question'],'answer':answer,'judgment':judgment}
                    result['answer_seconds']=answer_seconds
                except BudgetExceeded:raise
                except Exception as exc:result={'id':case['id'],'question':case['question'],'error':type(exc).__name__+': '+str(exc)}
                result.update(seconds=time.monotonic()-start,cost=gateway.spent-before,trace=trace)
                report.append(result);save(dest/('pilot-results.json' if args.phase=='run' else 'baseline-results.json'),report)
                print(json.dumps({k:v for k,v in result.items() if k not in ('trace','answer')},ensure_ascii=False),flush=True)
    finally:
        await db.close()
        print(json.dumps({'phase':args.phase,'cost_usd':gateway.spent,'completed':len(report)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['setup','stage','references','run','baseline']);parser.add_argument('--data-dir',required=True);parser.add_argument('--budget',type=float,default=3)
    args=parser.parse_args()
    if args.phase=='setup':setup(Path(args.data_dir).resolve())
    else:asyncio.run(execute(args))
