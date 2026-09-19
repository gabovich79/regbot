"""Offline diagnostic report. Never turns fallible pilot judgments into acceptance."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(rows, references):
    cases = []
    for row in rows:
        reference = references[row['id']]
        judge = row.get('judgment', {})
        reference_ok = (reference.get('literal_quotes_valid') is True and
                        reference.get('independent_check', {}).get('accepted') is True)
        schema_ok = (type(judge.get('correct')) is bool and type(judge.get('complete')) is bool and
                     all(isinstance(judge.get(k), list) and all(isinstance(v, str) for v in judge[k])
                         for k in ('unsupported_claims', 'missing_claims', 'notes')))
        cases.append({
            'id': row['id'], 'status': row.get('answer', {}).get('status', 'legacy_unverified'),
            'error': row.get('error'), 'answer_seconds': row.get('answer_seconds'),
            'cost_usd': row['cost'], 'reference_check_passed': reference_ok,
            'judge_schema_valid': schema_ok,
            'judge_suggestion': {k: judge[k] for k in ('correct', 'complete')} if schema_ok and reference_ok else None,
            'review_required': True,
        })
    times = sorted(c['answer_seconds'] for c in cases if c['answer_seconds'] is not None)
    return {'completed': len(cases), 'errors': sum(bool(c['error']) for c in cases),
            'status_counts': dict(Counter(c['status'] for c in cases if not c['error'])),
            'cost_usd_including_reservations': sum(c['cost_usd'] for c in cases),
            'answer_p95_nearest_rank_seconds': times[math.ceil(.95 * len(times)) - 1] if times else None,
            'latency_sample_size': len(times), 'cases': cases}


def report(directory):
    references = {r['id']: r for r in load(directory / 'references.json')}
    files = ['baseline-results.json', 'pilot-round0-results.json', 'pilot-round1-results.json', 'pilot-round2-results.json']
    runs = {name: load(directory / name) for name in files if (directory / name).exists()}
    if not runs:
        raise ValueError('No completed, archived pilot runs')
    summary = {'acceptance_passed': False, 'accuracy_percentage': None,
               'limitations': ['Development diagnostic only; no professional reference approval.',
                               'Legacy corpus has 36 active documents; pilot has 10 selected originals.',
                               'No web supplementation or conversation history in this pilot.',
                               'Judge suggestions require source review; missing/invalid fields are not passes.',
                               'Latency uses at most 10 questions per run and excludes external judging.'],
               'input_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ['references.json', 'source-receipt.json', *runs]},
               'runs': {name: summarize(rows, references) for name, rows in runs.items()}}
    (directory / 'pilot-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# RegBot — תיק סקירה של אבחון הפיתוח', '',
             '**לא מבחן קבלה ולא אישור מקצועי. לא חושב אחוז דיוק מאומת.**', '',
             'המסלול הישן חיפש ב־36 מסמכים; החדש בעשרה מסמכי מקור שנבחרו לפיילוט. רשת והיסטוריית שיחה הושבתו.', '',
             'זמני התשובה אינם כוללים את הבודק החיצוני. סטטוס ״נתמך״ הוא תוצאת בדיקת המערכת בלבד.', '']
    for key, reference in references.items():
        lines += [f"## {key}: {reference['question']}", '',
                  '### ייחוס שנוצר ונבדק אוטומטית — מחייב סקירה', '',
                  '```json', json.dumps(reference, ensure_ascii=False, indent=2), '```', '']
        for name, rows in runs.items():
            row = next((r for r in rows if r['id'] == key), None)
            if row is None:
                continue
            lines += [f'### {name}', '', f"זמן תשובה: {row.get('answer_seconds', 'לא הושלמה')} שניות.", '',
                      row.get('answer', {}).get('text', row.get('error', 'אין תשובה')), '',
                      'הערת בודק אוטומטי — אינה הכרעה:', '```json',
                      json.dumps(row.get('judgment', {}), ensure_ascii=False, indent=2), '```', '']
    (directory / 'pilot-review.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.data_dir), ensure_ascii=False))
