"""Verify verbatim unit composition on saved evidence, without retrieval retries.

The previous answer and expected answers are never provider inputs. This is a
composition diagnostic, not an end-to-end or held-out acceptance measurement.
"""
import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import time


async def main(args):
    import truststore
    truststore.inject_into_ssl()
    os.environ.update(DATA_DIR=str(args.data.resolve()), CAMPAIGN_BUDGET_DB=str(args.ledger.resolve()),
                      CAMPAIGN_BUDGET_USD='20', PROVIDER_PRICES_JSON=args.prices.read_text(encoding='utf-8'))
    from services import evidence_pipeline as pipeline
    from services.providers import Gateway
    args.output.mkdir(parents=True, exist_ok=False)
    for file in args.inputs:
        previous=json.loads(file.read_text(encoding='utf-8'))
        saved=previous['trace']
        if 'extracted_units_raw' not in saved:
            raise ValueError('A complete saved extraction is required')
        meter=Gateway(purpose='unit_composition_replay', limit=.5)
        class Replay:
            async def json(self, stage, payload, **kwargs):
                frozen={'understand':'understanding_raw', 'coverage':'coverage', 'evidence_units':'extracted_units_raw'}
                if stage in frozen:
                    return deepcopy(saved[frozen[stage]])
                return await meter.json(stage, payload, **kwargs)
        async def retrieve(db, plan, gateway, trace):
            for field in ('partial_section_seeds','omitted_section_seeds','unresolved_section_references'):
                trace[field]=deepcopy(saved.get(field, []))
            return deepcopy(saved['final_evidence'])
        pipeline.retrieve=retrieve
        trace={}
        result={'case':previous['case'], 'question':previous['question'], 'acceptance':False,
                'saved_input':str(file.resolve()), 'retrieval_provider_calls':0,
                'previous_status':previous.get('answer',{}).get('status')}
        started=time.monotonic()
        try:
            result['answer']=await asyncio.wait_for(pipeline.run_pipeline(previous['question'],[],None,Replay(),trace,
                                            enable_web=False,compose_units=True),90)
        except Exception as exc:
            result['error_type']=type(exc).__name__
        result.update(seconds=time.monotonic()-started, cost=meter.spent, calls=meter.calls, trace=trace)
        (args.output/file.name).write_text(json.dumps(result, ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'case':result['case'],'seconds':result['seconds'],'cost':result['cost'],
                          'status':result.get('answer',{}).get('status'),'error':result.get('error_type')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','ledger','prices','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    asyncio.run(main(parser.parse_args()))
