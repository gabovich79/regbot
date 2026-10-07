"""Build an offline, unapproved diagnostic reference bundle from exact originals.

No provider calls, DB writes, index activation, or runtime answer imports. The
annotations are development fixtures, never a locked acceptance set or legal gold.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('rule', 'scope', 'conditions', 'exceptions', 'period')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_bundle(spec, sources):
    cases, evidence, seen = [], {}, set()
    for case in spec['cases']:
        if case['id'] in seen:
            raise ValueError('Duplicate diagnostic case')
        seen.add(case['id'])
        selected = []
        for selection in case['sources']:
            source = sources[selection['document_id']]
            page_map = {p['page_number']: p['text'] for p in source['pages']}
            for page in selection['pages']:
                text = page_map[page]
                if not text.strip():
                    raise ValueError('Empty reference evidence')
                key = f"D{selection['document_id']}-V{source['original_sha256']}-X{digest(text)}-P{page if page is not None else 'DOCX'}"
                evidence[key] = {
                    'id': key, 'document_id': selection['document_id'],
                    'original_sha256': source['original_sha256'],
                    'title': source['title'], 'source_ref': source['source_ref'],
                    'page': page, 'char_start': 0, 'char_end': len(text),
                    'quote': text, 'quote_sha256': digest(text),
                    'extraction_note': 'Extracted text; DOCX equations linearized and explicit deletions annotated. Original bytes remain authoritative.',
                    'locator_note': 'PDF page text' if page is not None else 'DOCX body in document order; no stable page locator',
                }
                selected.append(key)
        requirements = []
        for field in FIELDS:
            if not isinstance(case.get(field), list) or not case[field]:
                raise ValueError(f'Missing reference dimension: {field}')
            for n, item in enumerate(case[field]):
                if not isinstance(item.get('text'), str) or not item['text'].strip():
                    raise ValueError('Empty annotation')
                refs = [key for key in selected if evidence[key]['page'] in item['pages']]
                if not refs or set(item['pages']) != {evidence[key]['page'] for key in refs}:
                    raise ValueError('Unresolved annotation evidence')
                requirements.append({'id': f"{case['id']}:{field}:{n}", 'kind': field,
                                     'text': item['text'], 'evidence_ids': refs})
        cases.append({'id': case['id'], 'question': case['question'],
                      'split': 'development_diagnostic', 'professional_approval': False,
                      'release_eligible': False, 'requirements': requirements,
                      'review_blockers': case['review_blockers'],
                      'technical_resolutions': case.get('technical_resolutions', []),
                      'expected_behavior': case.get('expected_behavior', 'source_scoped_answer'),
                      'evidence_ids': selected})
    return {'schema_version': 1, 'status': 'draft_pending_professional_review',
            'annotation_method': 'assistant_authored_from_original_extractions; not professionally approved',
            'is_acceptance_set': False, 'current_law_verified': False,
            'annotation_sha256': digest(json.dumps(spec, ensure_ascii=False, sort_keys=True)),
            'cases': cases, 'evidence': list(evidence.values())}


def load_originals(db_path, manifest, ids):
    sys.path.insert(0, str(ROOT))
    from services.document_service import extract_pdf_pages, extract_docx
    con = sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    result = {}
    try:
        for doc_id in sorted(ids):
            row = con.execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
            if row is None or not row['original_path']:
                raise ValueError(f'Original missing for D{doc_id}')
            original = Path(row['original_path'])
            actual = hashlib.sha256(original.read_bytes()).hexdigest()
            if actual != manifest[doc_id]['sha256']:
                raise ValueError(f'Original version mismatch for D{doc_id}')
            if original.suffix.lower() == '.pdf':
                pages = extract_pdf_pages(str(original))
            elif original.suffix.lower() == '.docx':
                pages = [{'page_number': None, 'text': extract_docx(str(original))}]
            else:
                raise ValueError('Unsupported reference format')
            result[doc_id] = {'original_sha256': actual, 'pages': pages,
                              'title': row['title'], 'source_ref': row['source_ref']}
    finally:
        con.close()
    return result


def markdown(bundle):
    labels = dict(rule='הכלל', scope='תחולה ואוכלוסייה', conditions='תנאים', exceptions='חריגים', period='תקופה')
    lines = ['# RegBot טבלת ייחוס לבדיקת תנאים וחריגים', '',
             'עשרה מקרי אבחון לפיתוח, המבוססים על תשעה קובצי מקור שזוהו לפי SHA-256. זו טיוטה לסקירה מקצועית, לא סט קבלה ולא אישור לתוקף הדין כיום.', '',
             'כל דרישה מקושרת לטקסט שחולץ מהמקור בקובץ JSON המצורף, כולל גרסת מקור, עמוד או גוף DOCX וטביעת תוכן. משוואות Word מוצגות בכתיב ליניארי ומחיקות מסומנות במפורש; הקובץ המקורי נשאר הסמכות. הסיכומים הם פרשנות לבדיקה.', '',
             'הקוד שמייצר תשובות אינו קורא את הטבלה. אין כאן תשובות מקודדות למערכת, ואין ציון הצלחה משפטי אוטומטי.', '']
    ev = {e['id']: e for e in bundle['evidence']}
    for case in bundle['cases']:
        lines += [f"## {case['id']} — {case['question']}", '']
        for requirement in case['requirements']:
            locations = ', '.join(f"D{ev[i]['document_id']} / " + (f"עמוד {ev[i]['page']}" if ev[i]['page'] is not None else 'גוף DOCX') for i in requirement['evidence_ids'])
            lines.append(f"- **{labels[requirement['kind']]}:** {requirement['text']} ({locations})")
        lines += ['', '**נותר לבדיקה:** ' + ' '.join(case['review_blockers']), '']
        if case['technical_resolutions']:
            lines += ['**בירורים טכניים שהושלמו:** ' + ' '.join(case['technical_resolutions']), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT/'eval/diagnostic-reference-spec.json').read_text(encoding='utf-8'))
    manifest = {d['id']: d for d in json.loads((ROOT/'eval/pilot-source-manifest.json').read_text(encoding='utf-8'))}
    ids = {s['document_id'] for c in spec['cases'] for s in c['sources']}
    bundle = build_bundle(spec, load_originals(args.db, manifest, ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'RegBot-reference-review-2026-09-19.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir/'RegBot-reference-review-2026-09-19.md').write_text(markdown(bundle), encoding='utf-8')
    print(json.dumps({'cases': len(bundle['cases']), 'source_files': len(ids), 'literal_evidence_spans': len(bundle['evidence']), 'release_eligible': False}))


if __name__ == '__main__':
    main()
