"""Readable local review form. Exported opinions never activate an index."""
import argparse
import html
import json
from pathlib import Path


def render(bundle, supplemental=None):
    esc = html.escape
    labels = {'rule': 'הכלל', 'scope': 'תחולה', 'conditions': 'תנאים', 'exceptions': 'חריגים', 'period': 'תקופה'}
    parts = ['''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RegBot — סקירה מקצועית של עשר תשובות ייחוס</title>
<style>body{font:18px/1.7 Arial,sans-serif;max-width:1000px;margin:auto;padding:28px;color:#17212b;background:#fff}h1{font-size:30px}h2{font-size:23px}section{border-top:2px solid #d9dfe6;margin-top:35px;padding-top:15px}p{margin:.7em 0}a{color:#155b91}summary{cursor:pointer;color:#155b91}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.65 Arial;background:#f4f6f8;padding:18px}select,textarea,button{font:inherit;padding:9px;max-width:100%;box-sizing:border-box}textarea{display:block;width:100%;min-height:90px}label{display:block;margin-top:14px}button{cursor:pointer;margin:15px 0}small{color:#425467}nav a{display:block}@media print{button,select,textarea{display:none}section{break-inside:avoid}}</style>
<h1>סקירה מקצועית של עשר תשובות ייחוס</h1>
<p>בדוק לכל שאלה אם הכלל, התנאים והחריגים להלן נכונים במסגרת המקור והתקופה המצוינים. אפשר לבחור ״מאשר — התשובה תקינה״, ״דורש תיקון״ או ״מחוץ לתחום המערכת״, ולהוסיף הערה. אישור התקינות מתייחס לנוסח הייחוס במסגרת המקור והתקופה המצוינים.</p>
<p><strong>זו סקירה של תשובות הייחוס, ולא של תשובות שהמערכת הפיקה בהרצה חדשה.</strong> אין כאן אישור לדין העדכני, לדיוק המערכת או לפריסה. במקרה הניוד נותר פער גרסה בתקנות, שמסומן בנפרד.</p>
<p>הטופס עובד מקומית. כדי לשמור את הבחירות וההערות לחץ בסיום על ״הורדת הערות הסקירה״ ושלח את הקובץ בצ׳אט. סגירת הדף ללא הורדה תאבד את ההערות. אפשר גם להשיב בצ׳אט לפי מספרי השאלות.</p>
<button type="button" onclick="exportReview()">הורדת הערות הסקירה</button><nav>''']
    for ordinal, case in enumerate(bundle['cases'], 1):
        parts.append(f'<a href="#q{ordinal}">{ordinal}. {esc(case["question"])}</a>')
    parts.append('</nav>')
    evidence = {e['id']: e for e in bundle['evidence']}
    for ordinal, case in enumerate(bundle['cases'], 1):
        parts.append(f'<section id="q{ordinal}"><h2>{ordinal}. {esc(case["question"])}</h2>')
        for field, label in labels.items():
            text = ' '.join(r['text'] for r in case['requirements'] if r['kind'] == field)
            parts.append(f'<p><strong>{label}:</strong> {esc(text)}</p>')
        parts.append('<p><strong>מגבלות ונקודות לסקירה:</strong> '+esc(' '.join(case['review_blockers']))+'</p>')
        if case.get('technical_resolutions'):
            parts.append('<details><summary>מה נבדק טכנית</summary><p>'+esc(' '.join(case['technical_resolutions']))+'</p></details>')
        parts.append('<details><summary>הראיות מהמסמכים המקוריים</summary>')
        for key in case['evidence_ids']:
            e = evidence[key]
            loc = f"עמוד {e['page']}" if e['page'] else 'גוף Word; המשוואות והמחיקות סומנו בחילוץ'
            parts.append(f'<details><summary>D{e["document_id"]} — {esc(e["title"])} — {loc}</summary>')
            if e.get('source_ref', '').startswith('https://'):
                parts.append(f'<a href="{esc(e["source_ref"], quote=True)}" target="_blank" rel="noopener noreferrer">פתיחת המקור</a>')
            parts.append(f'<pre>{esc(e["quote"])}</pre></details>')
        parts.append('</details>')
        if case['id'] == 'transfer' and supplemental:
            parts.append('<details><summary>ראיה משלימה מהרשות משנת 2025 — אינה תחליף לגרסת התקנות</summary><p>המכתב מצטט את התקנה ומורה על דחייה נקודתית בפברואר 2025. אין להסיק שהדחייה כללית או שזהו נוסח התקנות המלא.</p>')
            parts.append(f'<a href="{esc(supplemental["url"], quote=True)}" target="_blank" rel="noopener noreferrer">המקור הרשמי</a><pre>{esc(supplemental["quote"])}</pre></details>')
        parts.append(f'''<label for="decision{ordinal}">החלטתך לגבי נוסח הייחוס בלבד</label>
<select id="decision{ordinal}"><option value="pending">טרם נבדק</option><option value="source_scoped_correct">מאשר — התשובה תקינה</option><option value="needs_correction">דורש תיקון</option><option value="out_of_scope">מחוץ לתחום המערכת</option></select>
<label for="note{ordinal}">תיקון או הערה מקצועית</label><textarea id="note{ordinal}"></textarea></section>''')
    data = {'annotation_sha256': bundle['annotation_sha256'], 'scope': 'source_scoped_diagnostic_reference_review_only',
            'release_approval': False, 'cases': [{'id': c['id'], 'question': c['question']} for c in bundle['cases']]}
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    parts.append('<button type="button" onclick="exportReview()">הורדת הערות הסקירה</button>')
    parts.append('<script>const context='+encoded+''';
function exportReview(){const result={...context,reviewed_at:new Date().toISOString(),cases:context.cases.map((c,i)=>({...c,decision:document.getElementById('decision'+(i+1)).value,note:document.getElementById('note'+(i+1)).value}))};const u=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='RegBot-professional-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
</script></html>''')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--supplement', type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
    supplemental = json.loads(args.supplement.read_text(encoding='utf-8')) if args.supplement else None
    args.output.write_text(render(bundle, supplemental), encoding='utf-8')


if __name__ == '__main__':
    main()
