"""Validate evaluation evidence separately from runtime answer verification."""


def validate_judgment(judgment, case, answer, evidence):
    if not isinstance(judgment,dict):return False
    rows=judgment.get('requirements')
    expected={r['id'] for r in case['requirements']}
    if not isinstance(rows,list) or len(rows)!=len(expected):return False
    if any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):return False
    if {r['id'] for r in rows}!=expected:return False
    # Source lists and internal traces are not the answer a user read.
    body=answer.get('text','').split('\n\nמקורות ששימשו בתשובה:')[0]
    sources={e['id']:e['content'] for e in evidence}
    refusal=body.startswith('לא נמצאו ראיות מספיקות לתשובה מבוססת לשאלה זו.')
    for row in rows:
        if type(row.get('answered')) is not bool or type(row.get('retrieved')) is not bool:return False
        quotes=row.get('answer_quotes')
        spans=row.get('evidence_quotes')
        if not isinstance(quotes,list) or not isinstance(spans,list):return False
        if any(not isinstance(q,str) or not q.strip() or q not in body for q in quotes):return False
        if row['answered'] and (not quotes or refusal):return False
        for span in spans:
            if not isinstance(span,dict) or not isinstance(span.get('source_id'),str):return False
            source=sources.get(span['source_id'])
            quote=span.get('quote')
            if source is None or not isinstance(quote,str) or not quote.strip() or quote not in source:return False
        if row['retrieved'] and not spans:return False
    return all(isinstance(judgment.get(k),list) and all(isinstance(x,str) for x in judgment[k])
               for k in ('unsupported_claims','incorrect_claims','conflicts'))
