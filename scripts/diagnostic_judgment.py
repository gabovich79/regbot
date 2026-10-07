"""Validate evaluation evidence separately from runtime answer verification."""


def judgment_schema(case, answer, evidence):
    """Keep reference IDs distinct from pointers to observed runtime evidence."""
    from services.verification_protocol import obj
    answers, sources = judgment_spans(answer, evidence)
    def pointers(ids):
        # An empty pointer array remains valid; the sentinel is rejected by resolution.
        return {'type':'array','items':{'type':'string','enum':list(ids) or ['NONE']}}
    strings = {'type':'array','items':{'type':'string'}}
    row = obj({'id':{'type':'string','enum':[r['id'] for r in case['requirements']]},
               'retrieved':{'type':'boolean'},'answered':{'type':'boolean'},
               'reason':{'type':'string'},'answer_span_ids':pointers(answers),
               'evidence_ids':pointers(sources)})
    return obj({'requirements':{'type':'array','items':row},
                'unsupported_claims':strings,'incorrect_claims':strings,'conflicts':strings})


def judgment_spans(answer, evidence):
    body=answer.get('text','').split('\n\nמקורות ששימשו בתשובה:')[0]
    answer_spans={f'A{i+1}':text for i,text in enumerate(line.strip() for line in body.splitlines() if line.strip())}
    evidence_spans={f'E{i+1}':{'source_id':e['id'],'quote':e['content']} for i,e in enumerate(evidence)}
    return answer_spans,evidence_spans


def resolve_judgment(raw, answer, evidence):
    """Resolve evaluator pointers, never its rewritten quotations."""
    from copy import deepcopy
    result=deepcopy(raw)
    answers,sources=judgment_spans(answer,evidence)
    rows=result.get('requirements')
    if not isinstance(rows,list):raise ValueError('Missing evaluation requirements')
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid evaluation row')
        a=row.get('answer_span_ids');e=row.get('evidence_ids')
        if not isinstance(a,list) or any(not isinstance(i,str) or i not in answers for i in a):raise ValueError('Unknown answer span')
        if not isinstance(e,list) or any(not isinstance(i,str) or i not in sources for i in e):raise ValueError('Unknown retrieved span')
        row['answer_quotes']=[answers[i] for i in dict.fromkeys(a)]
        row['evidence_quotes']=[sources[i] for i in dict.fromkeys(e)]
    return result


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
