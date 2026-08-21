import fitz

from services.document_service import extract_pdf_pages


def test_extract_pdf_pages_preserves_page_numbers_and_text(tmp_path):
    pdf_path = tmp_path / "two-pages.pdf"
    source = fitz.open()
    try:
        first = source.new_page()
        first.insert_text((72, 72), "First regulatory page")
        second = source.new_page()
        second.insert_text((72, 72), "Second regulatory page")
        source.save(pdf_path)
    finally:
        source.close()

    pages = extract_pdf_pages(str(pdf_path))

    assert pages == [
        {"page_number": 1, "text": "First regulatory page"},
        {"page_number": 2, "text": "Second regulatory page"},
    ]
