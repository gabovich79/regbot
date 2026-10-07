"""Replay frozen ranked candidates without provider calls or index mutations."""
import argparse
import asyncio
import json
import os
from pathlib import Path


async def main(args):
    os.environ['DATA_DIR']=str((args.source/'runtime').resolve())
    from models.database import get_db
    from models.evidence_store import version_chunks
    from services.section_context import pack_sections
    from services.evidence_search import public_evidence
    freeze=json.loads((args.source/'freeze.json').read_text(encoding='utf-8'))
    db=await get_db()
    try:
        chunks=await version_chunks(db,list(freeze['manifest'].values()),approved_only=False)
    finally:
        await db.close()
    lookup={c['id']:c for c in chunks}
    results=[]
    for path in sorted((args.source/'results').glob('*-current.json')):
        original=json.loads(path.read_text(encoding='utf-8'))
        ids=original['trace'].get('reranked_ids',[])
        if not ids:
            continue
        evidence,trace=pack_sections([lookup[i] for i in ids],chunks,public_evidence)
        assert len({e['id'] for e in evidence})==len(evidence)
        assert trace['context_tokens']<=24000
        results.append({'case':original['id'],'old_count':len(original['trace'].get('final_evidence',[])),
                        'new_count':len(evidence),'selection':trace,'evidence':evidence,
                        'new_ids':[e['id'] for e in evidence]})
    args.output.write_text(json.dumps({'acceptance':False,'provider_calls':0,'cases':results},ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:
        print(json.dumps({'case':r['case'],'old':r['old_count'],'new':r['new_count'],
                          'tokens':r['selection']['context_tokens'],
                          'partial':len(r['selection']['partial_section_seeds'])}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
