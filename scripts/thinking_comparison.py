"""Paired synthesis-only experiment on new formulations, with fixed sources.

These are new formulations of development topics, NOT unseen acceptance cases.
Reference requirements are never read. No model configuration is promoted here.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

CASES = {
    'fees':'בהסדר שבו הובטחו לעמית דמי ניהול מוזלים, מתי אפשר לשנות אותם כלפי מעלה, מה צריך להודיע, ועל אילו הסדרים הכללים אינם חלים? התייחס לנוסח המצורף.',
    'employer':'פרט לפי החוזר המצורף את דיווחי המעסיק ואת התשובות שהוא אמור לקבל מהקופה: מה נמסר, באילו מועדים, והאם יש פטור מותנה למעסיק קטן?',
    'annual-cost':'לפי המסמך המצורף, כיצד מחברים את מרכיבי העלות השנתית הצפויה, מאיזו שנה נלקח כל רכיב, ואילו מוצרים או מסלולים דורשים טיפול שונה?',
    'loan':'לפי הנוסח המצורף, השווה מתן הלוואה לעמית מקרן השתלמות נזילה לעומת קרן שאינה נזילה. פרט תקרות, משך, שיקול דעת הגוף ותנאי החזר וחריגיהם.'
}


async def main(args):
    import truststore
    truststore.inject_into_ssl()
    os.environ.update(DATA_DIR=str((args.source/'runtime').resolve()),
        CAMPAIGN_BUDGET_DB=str(args.ledger.resolve()),CAMPAIGN_BUDGET_USD='20',
        PROVIDER_PRICES_JSON=(args.source/'prices.json').read_text(encoding='utf-8'))
    from services.providers import Gateway
    from scripts.decision_comparison import TASK,source_only,quote_checks
    args.output.mkdir(parents=True,exist_ok=False)
    metadata={'cases':CASES,'budgets':[0,1024],'acceptance':False,'source_hashes':{},'max_output':8192,'timeout':90}
    for case in CASES:
        data=(args.source/f'oracle-{case}.json').read_bytes()
        metadata['source_hashes'][case]=hashlib.sha256(data).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    for case,question in CASES.items():
        evidence=json.loads((args.source/f'oracle-{case}.json').read_text(encoding='utf-8'))
        # Identical evidence and prompt in both arms. No expected answer input.
        payload={'task':TASK,'question':question,'evidence':[source_only(e) for e in evidence]}
        for budget in metadata['budgets']:
            gateway=Gateway(purpose='thinking_comparison',limit=.5)
            result={'case':case,'question':question,'thinking_budget':budget,'payload':payload,'correctness':'ungraded'}
            started=time.monotonic()
            try:
                answer=await asyncio.wait_for(gateway.json('direct_synthesis',payload,max_output=8192,thinking_budget=budget),90)
                result.update(answer=answer,provenance_issues=quote_checks(answer,evidence))
            except Exception as exc:
                result['error_type']=type(exc).__name__
            result.update(seconds=time.monotonic()-started,cost=gateway.spent,calls=gateway.calls)
            (args.output/f'{case}-{budget}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:result.get(k) for k in ('case','thinking_budget','seconds','cost','error_type')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','output','ledger'):
        parser.add_argument('--'+arg,type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
