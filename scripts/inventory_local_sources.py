"""Read-only local source inventory; candidate matches are not approval to ingest."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.document_service import extract_pdf_pages, extract_docx_bytes
from services.document_integrity_service import assess_document_integrity
from services.document_profile_service import build_document_profile


def normalized(text):
    return re.sub(r'\s+', '', text)


def shingles(text):
    words = re.findall(r'[\w\u0590-\u05ff]+', text.lower())
    return {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}


def inventory(source, reference, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    old = []
    for item in manifest:
        path = reference/f"{item['id']}.txt"
        if path.exists():
            text = path.read_text(encoding='utf-8', errors='replace')
            old.append((item, normalized(text), shingles(text)))
    records = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.pdf', '.docx'):
            continue
        blob = path.read_bytes()
        checksum = hashlib.sha256(blob).hexdigest()
        record = {'file': str(path), 'sha256': checksum, 'bytes': len(blob)}
        try:
            pages = extract_pdf_pages(str(path)) if path.suffix.lower()=='.pdf' else []
            text = '\n\n'.join(p['text'] for p in pages) if pages else extract_docx_bytes(blob)
            (output/f'{checksum}.txt').write_text(text,encoding='utf-8')
            if pages:
                (output/f'{checksum}.pages.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
            record.update(chars=len(text), pages=len(pages) or None,
                          empty_pages=[p['page_number'] for p in pages if not p['text'].strip()],
                          excerpt=text[:1200])
            profile = build_document_profile({'id':0,'title':path.stem},text)
            record['integrity'] = assess_document_integrity({'title':path.stem},text,profile)
            record['canonical_title'] = profile['canonical_title']
            record['draft_marker_in_opening'] = 'טיוטה' in text[:3000] or 'draft' in path.name.lower()
            norm, grams = normalized(text), shingles(text)
            matches = []
            for item, prior, prior_grams in old:
                union = grams | prior_grams
                score = len(grams & prior_grams)/len(union) if union else 0
                exact = bool(norm) and norm==prior
                matches.append({'id':item['id'], 'title':item['title'],
                    'exact_normalized_text':exact, 'shingle_jaccard':round(score,4),
                    'original_already_retained':bool(item.get('original_path'))})
            record['candidates'] = sorted(matches,key=lambda m:(m['exact_normalized_text'],m['shingle_jaccard']),reverse=True)[:3]
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        records.append(record)
    result={'source_directory':str(source),'reference':'historical extraction snapshots; confirm hashes against live snapshot before attaching', 'files':records}
    (output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=inventory(args.source,args.reference,args.output)
    for row in result['files']:
        print(json.dumps({k:row.get(k) for k in ('file','pages','chars','integrity','draft_marker_in_opening','candidates','error')},ensure_ascii=False))
