import fitz

from services.document_service import extract_pdf_bytes_pages
from services.knowledge import quality_issues


def test_ruled_table_preserves_text_but_requires_structure_review():
    doc = fitz.open()
    page = doc.new_page()
    for x in (50, 200, 350):
        page.draw_line((x, 50), (x, 150))
    for y in (50, 100, 150):
        page.draw_line((50, y), (350, y))
    for x, y, text in ((60, 75, 'Item'), (210, 75, 'Requirement'),
                       (60, 125, 'Notification'), (210, 125, 'Recipient address')):
        page.insert_text((x, y), text)
    pages = extract_pdf_bytes_pages(doc.tobytes())
    doc.close()
    assert 'Recipient address' in pages[0]['text']
    assert pages[0]['table_detection_status'] == 'completed'
    assert pages[0]['table_regions'][0]['columns'] == 2
    assert 'pdf_tables_require_structure_review' in quality_issues(pages[0]['text'], pages)


def test_table_detection_failure_is_not_silently_approved(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('unreadable vector geometry')
    monkeypatch.setattr(fitz.Page, 'find_tables', fail)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), 'A readable paragraph with no preserved table geometry.')
    pages = extract_pdf_bytes_pages(doc.tobytes())
    doc.close()
    assert pages[0]['text'].startswith('A readable paragraph')
    assert 'pdf_table_detection_failed_requires_review' in quality_issues(pages[0]['text'], pages)
