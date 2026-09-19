"""Evidence-aware acceptance gates, separate from answer generation."""
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def gold_ready(case):
    if case.get('review_status') not in ('machine_checked_pending_human','human_verified'):
        return False
    references = case.get('references',[])
    if not case.get('expected_abstention') and not references:
        return False
    return all(r.get('quote') and r.get('url') and r.get('source_hash') and r.get('quote_verified') is True for r in references)


def score_trace(case, trace, judgment):
    evidence = trace.get('corpus_evidence',trace.get('final_evidence',[]))
    quotes = [r['quote'] for r in case.get('references',[]) if r.get('required_for_retrieval')]
    # Exact evidence spans, not document titles or auto-appended footers.
    found = sum(any(q in e['content'] for e in evidence if e['kind']=='corpus') for q in quotes)
    checks = {key:judgment.get(key) is True for key in ('correct','complete','supported','handles_missing','handles_conflicts')}
    critical = judgment.get('critical_error') is not False
    return {'id':case['id'], 'gold_ready':gold_ready(case), 'required_units':len(quotes), 'retrieved_units':found,
            'passed':gold_ready(case) and all(checks.values()) and not critical,
            'critical_error':critical,'checks':checks,'status':trace.get('status'),
            'web_used':bool(trace.get('web_evidence')), 'response_time_ms':trace.get('response_time_ms',0),
            'failure_stage':judgment.get('failure_stage','unknown')}


def release_gate(runs, human_approved=False):
    reasons=[]
    if len(runs)!=3:
        reasons.append('three_complete_runs_required')
    fingerprints={r.get('case_fingerprint') for r in runs}
    identities={tuple(sorted(c['id'] for c in r.get('cases',[]))) for r in runs}
    if len(fingerprints)!=1 or None in fingerprints or len(identities)!=1:
        reasons.append('runs_must_use_identical_frozen_cases')
    for run in runs:
        cases=run.get('cases',[])
        if len(cases)!=20 or len({c['id'] for c in cases})!=20 or not all(c['gold_ready'] for c in cases):
            reasons.append('20_frozen_reviewable_cases_required')
        if not cases or sum(c['passed'] for c in cases)/len(cases)<.90:
            reasons.append('answer_quality_below_90_percent')
        if any(c['critical_error'] for c in cases):
            reasons.append('critical_error')
        if any(not c['checks']['handles_missing'] or not c['checks']['handles_conflicts'] for c in cases):
            reasons.append('missing_or_conflict_handling_failed')
        total=sum(c['required_units'] for c in cases)
        found=sum(c['retrieved_units'] for c in cases)
        if not total or found/total < .95:
            reasons.append('corpus_evidence_recall_below_95_percent')
        times=sorted(c['response_time_ms'] for c in cases)
        if not times or times[max(0,__import__('math').ceil(.95*len(times))-1)]>60000:
            reasons.append('latency_target_failed')
    if not human_approved:
        reasons.append('human_approval_pending')
    return {'release_ready':not reasons,'blockers':sorted(set(reasons))}
