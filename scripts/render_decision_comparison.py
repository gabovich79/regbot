"""Render ungraded paired results with source-backed expectations, offline."""
import argparse
import html
import json
import statistics
from pathlib import Path


def esc(value):
    return html.escape(str(value))


def render(root, destination):
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    bundle = read(root/'references.json')
    names = {'current':'המנוע הנוכחי','simple':'חיפוש וסינתזה קצרה','oracle':'מקורות ידועים מראש'}
    sections, metrics = [], []
    for case in bundle['cases']:
        cards = []
        for arm, label in names.items():
            path = root/'results'/f"{case['id']}-{arm}.json"
            if not path.exists():
                cards.append(f'<article><h3>{label}</h3><p>טרם הושלם</p></article>')
                continue
            result = read(path)
            answer = result.get('answer',{})
            text = answer.get('text') or '\n\n'.join(c.get('text','') for c in answer.get('claims',[]))
            if answer.get('clarification'):
                text += '\n\n'+answer['clarification']
            if answer.get('missing'):
                text += '\n\nמידע חסר:\n'+'\n'.join(answer['missing'])
            trace = result.get('trace',{})
            issue_count = len(trace.get('citation_provenance_issues',[]))
            evidence = trace.get('final_evidence',[])
            metrics.append({'id':case['id'],'arm':arm,'seconds':round(result['seconds'],2),'cost':result['cost'],
                'error':result.get('error_type'),'characters':len(text),'evidence_items':len(evidence),
                'provenance_issues':issue_count if arm != 'current' else None,'correctness':'ungraded'})
            source_html = ''.join(f"<details><summary>{esc(e.get('title',''))} — {esc(e.get('page_start'))}</summary><small>{esc(e['id'])}</small><pre>{esc(e['content'])}</pre></details>" for e in evidence)
            claims_html = ''
            for claim in answer.get('claims',[]):
                claims_html += '<details><summary>'+esc(claim.get('text',''))+'</summary>'
                for citation in claim.get('citations',[]):
                    claims_html += '<small>'+esc(citation.get('id',''))+'</small><blockquote>'+esc(citation.get('quote',''))+'</blockquote>'
                claims_html += '</details>'
            cards.append(f"<article><h3>{label}</h3><p>{result['seconds']:.1f} שניות · ${result['cost']:.4f} · {len(text):,} תווים</p><p>נכונות: טרם דורגה. שגיאה: {esc(result.get('error_type') or 'אין שגיאת הרצה')}</p><pre>{esc(text)}</pre><details><summary>טענות וציטוטים</summary>{claims_html}</details><details><summary>הראיות שנמסרו ({len(evidence)})</summary>{source_html}</details></article>")
        expected = ''.join('<li>'+esc(r['text'])+'</li>' for r in case['requirements'])
        sections.append(f"<section><h2>{esc(case['question'])}</h2><details><summary>הדרישות שאושרו ביחס למקורות</summary><ul>{expected}</ul><p>האישור מתייחס למקורות השמורים ולמגבלות שנרשמו, לא לתשובות המערכת או לדין כיום.</p></details><div class='grid'>{''.join(cards)}</div></section>")
    findings = ''
    overview = '<section><h2>תוצאות ההרצה — אינן ציוני נכונות</h2><table><tr><th>מסלול</th><th>הושלמו ללא שגיאת הרצה</th><th>חציון זמן של תשובות שהושלמו</th><th>עלות ושמירות</th></tr>'
    for arm,label in names.items():
        rows = [m for m in metrics if m['arm']==arm]
        completed = [m for m in rows if not m['error']]
        median = f"{statistics.median(m['seconds'] for m in completed):.1f} שניות" if completed else '—'
        overview += f"<tr><td>{label}</td><td>{len(completed)} / {len(rows)}</td><td>{median}</td><td>${sum(m['cost'] for m in rows):.4f}</td></tr>"
    overview += '</table><p>כשלי ספק נשמרו ללא ניסיון חוזר. העלות כוללת יתרות שנשמרו לבקשות שנכשלו, עד בירור החיוב.</p></section>'
    if (root/'review-findings.json').exists():
        review = read(root/'review-findings.json')
        findings = '<section><h2>ממצאים שנבדקו מול המקורות</h2><p>סקירת כשלים ממוקדת; אינה דירוג מלא או אישור מקצועי.</p>'
        for finding in review['findings']:
            findings += '<h3>'+esc(finding['title'])+'</h3><p>'+esc(finding['finding'])+'</p>'
            if finding.get('source_quote'):
                findings += '<blockquote>'+esc(finding['source_quote'])+'</blockquote>'
        findings += '</section>'
    destination.write_text("<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>RegBot — השוואת אבחון</title><style>body{font:17px system-ui;margin:24px;background:#f5f7fa;color:#172536}h1,h2{line-height:1.5}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}article,section{background:white;padding:18px;border:1px solid #d9e0e8;border-radius:9px;margin-bottom:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.7}.grid>article>pre{max-height:480px;overflow:auto}td,th{padding:12px;text-align:right;border-bottom:1px solid #ddd}small{overflow-wrap:anywhere}summary{cursor:pointer;padding:8px}blockquote{border-right:3px solid #4388aa;padding:12px} @media(max-width:1000px){.grid{grid-template-columns:1fr}}</style><h1>RegBot — השוואת אבחון, 7.10.2026</h1><p>אותן עשר שאלות ומקורות קפואים. ללא חיפוש רשת. זהו מבחן אבחון ולא מבחן קבלה. סטטוס אוטומטי וציטוט שקיים במסמך אינם הוכחת נכונות.</p><p>מסלול המקורות הידועים מקבל עמודים ומסמכי Word שלמים, עד 64,000 טוקנים; שני המסלולים האחרים עד 24,000. לכן הוא בודק יכולת הבנה כאשר המקור זמין, ולא יעילות של מנוע חלופי. שינויי חילוץ Word מתועדים בנפרד.</p>"+overview+findings+''.join(sections)+"</html>",encoding='utf-8')
    (root/'observations.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('destination',type=Path)
    args = parser.parse_args()
    render(args.root,args.destination)
