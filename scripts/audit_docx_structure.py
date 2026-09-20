"""Offline extraction inventory. This is not legal or visual source approval."""
import argparse
import hashlib
import json
from pathlib import Path

from docx import Document

from services.document_service import _docx_inline_text, clean_text, extract_docx


def audit(path):
    path = Path(path)
    doc = Document(path)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    body = doc.element.body
    text = extract_docx(str(path))
    paragraphs = [clean_text(_docx_inline_text(p)) for p in body.findall('.//w:p', ns)]
    return {
        'original_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'extracted_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'characters': len(text),
        'tables': len(body.findall('.//w:tbl', ns)),
        'physical_cells': len(body.findall('.//w:tc', ns)),
        'horizontal_spans': len(body.findall('.//w:gridSpan', ns)),
        'vertical_merge_cells': len(body.findall('.//w:vMerge', ns)),
        'explicit_numbered_paragraphs': len(body.findall('.//w:pPr/w:numPr', ns)),
        'source_paragraphs': len(paragraphs),
        'missing_paragraphs': [p for p in paragraphs if p and p not in text],
        'approved': False,
        'limitations': [
            'Whitespace-normalized paragraph presence does not establish cell association or multiplicity',
            'Automatic numbering, including numbering inherited from styles, is not reconstructed',
            'Tracked revisions and table layout require source review',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    result = audit(args.source)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
