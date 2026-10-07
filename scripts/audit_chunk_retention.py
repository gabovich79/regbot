"""Offline source retention diagnostic. This is not retrieval/answer acceptance."""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.document_service import extract_pdf_pages
from services.knowledge import prepare_document


def windows(text, size=8):
    words = re.findall(r'[\w\u0590-\u05ff]+',text.lower())
    return {tuple(words[i:i+size]) for i in range(max(0,len(words)-size+1))}


def audit(db_path):
    db=sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    results=[]
    try:
        for row in db.execute('SELECT * FROM documents WHERE is_active=1 AND original_path IS NOT NULL ORDER BY id'):
            document=dict(row)
            original=Path(document['original_path'])
            if not original.is_file() or original.suffix.lower()!='.pdf':
                continue
            pages=extract_pdf_pages(str(original))
            source=set().union(*(windows(p['text']) for p in pages))
            old=db.execute('SELECT content FROM document_chunks WHERE document_id=? ORDER BY chunk_index',(document['id'],)).fetchall()
            old_windows=set().union(*(windows(c['content']) for c in old))
            _,_,issues,new=prepare_document('',document,pages)
            new_windows=set().union(*(windows(c['content']) for c in new))
            missing=sorted(source-old_windows)
            results.append({'id':document['id'],'title':document['title'],
                'source_pages':len(pages),'extraction_issues':issues,'source_windows':len(source),
                'legacy_chunks':len(old),'candidate_chunks':len(new),
                'legacy_retained_windows':len(source & old_windows),
                'candidate_retained_windows':len(source & new_windows),
                'legacy_retention':len(source & old_windows)/len(source) if source else None,
                'candidate_retention':len(source & new_windows)/len(source) if source else None,
                'legacy_missing_samples':[' '.join(w) for w in missing[:12]]})
    finally:
        db.close()
    return {'metric':'unique eight-word windows within each extracted PDF page, present in at least one chunk',
            'limitations':'Measures retention of extracted text only, not PDF extraction completeness, legal correctness, retrieval recall or acceptance. Cross-section windows may be absent even when both sections are retained.',
            'documents':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.db)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([{k:d[k] for k in ('id','legacy_chunks','candidate_chunks','legacy_retention','candidate_retention','extraction_issues')} for d in result['documents']],ensure_ascii=False))
