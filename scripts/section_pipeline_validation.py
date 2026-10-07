"""One-shot end-to-end development validation of section context changes."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time


async def main(args):
    import truststore
    truststore.inject_into_ssl()
    os.environ.update(DATA_DIR=str((args.source/'runtime').resolve()),CAMPAIGN_BUDGET_DB=str(args.ledger.resolve()),
                      CAMPAIGN_BUDGET_USD='20',PROVIDER_PRICES_JSON=(args.source/'prices.json').read_text(encoding='utf-8'))
    from models.database import get_db
    from models.evidence_store import version_chunks
    from services import evidence_search as search
    from services.evidence_pipeline import run_pipeline
    from services.providers import Gateway
    from scripts.thinking_comparison import CASES
    freeze=json.loads((args.source/'freeze.json').read_text(encoding='utf-8'))
    async def selected(db):
        return 'UNAPPROVED-SECTION-VALIDATION',await version_chunks(db,list(freeze['manifest'].values()),approved_only=False)
    search.active_chunks=selected
    args.output.mkdir(parents=True,exist_ok=False)
    for case in ('fees','annual-cost'):
        question=CASES[case].replace('המצורף','שבמאגר')
        trace={};g=Gateway(purpose='section_pipeline_validation',limit=.5);db=await get_db();started=time.monotonic()
        result={'case':case,'question':question,'acceptance':False}
        try:
            result['answer']=await asyncio.wait_for(run_pipeline(question,[],db,g,trace,enable_web=False),90)
        except Exception as exc:
            result['error_type']=type(exc).__name__
        finally:
            await db.close()
        result.update(seconds=time.monotonic()-started,cost=g.spent,trace=trace,calls=g.calls)
        (args.output/f'{case}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'case':case,'seconds':result['seconds'],'cost':g.spent,'error':result.get('error_type'),
                          'status':result.get('answer',{}).get('status')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','output','ledger'):
        parser.add_argument('--'+arg,type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
