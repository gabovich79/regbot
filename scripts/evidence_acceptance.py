"""Build source-backed gold, freeze it, evaluate v2, and enforce release gates.

No candidate/reference_answer is treated as verified gold until literal source
quotes have been fetched and checked. Network stages require configured prices.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.acceptance import fingerprint, gold_ready, score_trace, release_gate


def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


async def prepare(args):
    from models.database import init_db
    from services.providers import Gateway
    from services.web_evidence import fetch_source
    await init_db()
    cases = read_cases(args.cases)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Refusing to overwrite a prepared reference set')
    snapshots = output.parent/(output.stem+'-sources')
    snapshots.mkdir(parents=True,exist_ok=True)
    gateway = Gateway(purpose='evaluation_reference',limit=args.budget)
    prepared=[]
    for case in cases:
        item = dict(case, references=[],review_status='pending_source_review')
        try:
            sources=[]
            for url in case.get('reference_urls',[]):
                source = await fetch_source(url)
                (snapshots/(source['source_hash']+'.json')).write_text(json.dumps(source,ensure_ascii=False),encoding='utf-8')
                sources.append(source)
            if not sources:
                raise ValueError('No approved source URLs: curator must supply references')
            result = await gateway.json('gold_reference',{
                'task':'Build a test reference answer FROM SOURCES, not from any bot answer. Return expected_answer, '
                       'required_claims [text], exceptions [text], expected_abstention boolean, '
                       'references [{url,quote,required_for_retrieval:boolean}]. Quotes must be exact contiguous source text. '
                       'Do not infer missing dates. Mark unresolved or non-covered questions explicitly.',
                'case':case,'sources':sources,
            })
            by_url={s['url']:s for s in sources}
            references=[]
            for ref in result.get('references',[]):
                source=by_url.get(ref.get('url'))
                quote=ref.get('quote')
                if not source or not isinstance(quote,str) or len(quote)<10 or not any(quote in p['content'] for p in source['pages']):
                    raise ValueError('Reference quote failed exact source verification')
                references.append({**ref,'source_hash':source['source_hash'],'quote_verified':True})
            if not references:
                raise ValueError('Gold preparation did not produce source evidence')
            check = await gateway.json('verify_gold',{
                'task':'Independently check that the reference answers the question correctly and completely from sources. '
                       'Return accepted boolean and issues [text]. Check conditions, exceptions and applicable years. '
                       'This does not constitute human approval.', 'question':case['question'],'reference':result,'sources':sources,
            })
            item.update(expected_answer=result.get('expected_answer'),required_claims=result.get('required_claims',[]),
                        exceptions=result.get('exceptions',[]),references=references,
                        expected_abstention=result.get('expected_abstention') is True,
                        reference_check=check)
            if check.get('accepted') is True and not check.get('issues'):
                item['review_status']='machine_checked_pending_human'
        except Exception as exc:
            item['preparation_error']=str(exc)
        prepared.append(item)
        # Durable progress; interrupted preparation never silently becomes gold.
        output.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in prepared),encoding='utf-8')
    print(json.dumps({'prepared':len(prepared),'reviewable':sum(gold_ready(c) for c in prepared),'cost_usd':gateway.spent}))


def freeze(args):
    cases=read_cases(args.cases)
    if len(cases)!=60 or len({c['id'] for c in cases})!=60 or not all(gold_ready(c) for c in cases):
        raise ValueError('Exactly 60 unique source-backed cases required before freeze')
    counts={split:sum(c['split']==split for c in cases) for split in ('development','acceptance')}
    if counts!={'development':40,'acceptance':20}:
        raise ValueError('Expected 40 development and 20 acceptance cases')
    groups={}
    for case in cases:
        group=case['scenario_group']
        if group in groups and groups[group]!=case['split']:
            raise ValueError('Scenario group leaks across splits')
        groups[group]=case['split']
    Path(args.output).write_text(json.dumps({'fingerprint':fingerprint(cases),'counts':counts},indent=2),encoding='utf-8')


async def run(args):
    import uuid
    import subprocess
    from datetime import datetime, timezone
    from config import DEFAULT_MODEL, EMBEDDING_MODEL
    from services.acceptance import summarize_cases
    from models.database import init_db,get_db
    from services.providers import Gateway
    from services.evidence_pipeline import run_pipeline
    await init_db()
    cases=read_cases(args.cases)
    lock=json.loads(Path(args.lock).read_text(encoding='utf-8'))
    if fingerprint(cases)!=lock['fingerprint']:
        raise ValueError('Frozen cases changed')
    cases=[c for c in cases if c['split']==args.split]
    gateway=Gateway(purpose='evaluation',limit=args.budget)
    report={'run_id':uuid.uuid4().hex,'started_at':datetime.now(timezone.utc).isoformat(),
            'split':args.split,'case_fingerprint':lock['fingerprint'],'cases':[],'details':[],'web_enabled':not args.no_web}
    db=await get_db()
    try:
        active=await (await db.execute('SELECT release_id FROM active_index WHERE singleton=1')).fetchone()
        if not active:
            raise ValueError('Acceptance requires an activated reviewed index')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
        if dirty:
            raise ValueError('Commit code changes before measuring a reproducible acceptance run')
        versions=await (await db.execute('SELECT id,source_hash,embedding_model,card,review_status FROM evidence_versions ORDER BY id')).fetchall()
        report['runtime']={'commit':commit,'active_index':active['release_id'],
            'generation_model':DEFAULT_MODEL,'embedding_model':EMBEDDING_MODEL,
            'index_state_hash':fingerprint([dict(v) for v in versions]),'web_enabled':not args.no_web}
        report['runtime_fingerprint']=fingerprint(report['runtime'])
        for case in cases:
            trace={}
            answer=None
            start=time.monotonic()
            try:
                answer=await asyncio.wait_for(run_pipeline(case['question'],case.get('history',[]),db,gateway,trace,enable_web=not args.no_web),90)
                trace['response_time_ms']=int((time.monotonic()-start)*1000)
                judgment=await gateway.json('acceptance_judge',{
                    'task':'Judge independently against the frozen reference. Return booleans correct,complete,supported,handles_missing,handles_conflicts,critical_error; '
                           'failure_stage one of source,extraction,chunking,retrieval,validity,synthesis,citation,none; reasons [text]. '
                           'A footnote alone never proves correctness. Abstention passes only if appropriate.',
                    'case':case,'answer':answer,'evidence':trace.get('final_evidence',[]),
                })
            except Exception as exc:
                trace['error']=str(exc)
                judgment={'critical_error':True,'failure_stage':'runtime'}
            report['cases'].append(score_trace(case,trace,judgment))
            report['details'].append({'id':case['id'],'question':case['question'],
                'reference':case,'answer':answer,'trace':trace,'judgment':judgment})
            report['summary']=summarize_cases(report['cases'])
            report['cost_usd']=gateway.spent
            Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        await db.close()
    print(json.dumps({'passed':sum(c['passed'] for c in report['cases']),'total':len(report['cases']),'cost':gateway.spent}))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','freeze','run'):
        p=sub.add_parser(command)
        p.add_argument('--cases',required=True)
        p.add_argument('--output',required=True)
        if command in ('prepare','run'):
            p.add_argument('--budget',type=float,required=True,help='Explicit maximum USD for this batch')
        if command=='run':
            p.add_argument('--lock',required=True)
            p.add_argument('--split',choices=['development','acceptance'],default='development')
            p.add_argument('--no-web',action='store_true')
    gate=sub.add_parser('gate')
    gate.add_argument('runs',nargs=3)
    gate.add_argument('--human-approved',action='store_true',help='Use only after recorded professional approval')
    args=parser.parse_args()
    if args.command=='prepare': asyncio.run(prepare(args))
    elif args.command=='freeze': freeze(args)
    elif args.command=='run': asyncio.run(run(args))
    else:
        result=release_gate([json.loads(Path(p).read_text(encoding='utf-8')) for p in args.runs],args.human_approved)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(0 if result['release_ready'] else 1)


if __name__=='__main__':
    main()
