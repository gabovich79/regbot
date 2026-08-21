import fitz

from services.document_service import extract_pdf_bytes_pages, extract_pdf_pages
from services.rag_service import chunk_regulatory_pages


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


def test_pdf_bytes_and_page_chunks_keep_page_provenance():
    source = fitz.open()
    try:
        first = source.new_page()
        first.insert_text((72, 72), "First page regulatory content with enough text to form a chunk.")
        second = source.new_page()
        second.insert_text((72, 72), "Second page regulatory content with enough text to form a chunk.")
        pdf_bytes = source.tobytes()
    finally:
        source.close()

    pages = extract_pdf_bytes_pages(pdf_bytes)
    chunks = chunk_regulatory_pages(
        pages, {"id": 9, "title": "Document", "source_ref": "https://example.test/doc.pdf"}
    )

    assert [page["page_number"] for page in pages] == [1, 2]
    assert [(chunk["page_start"], chunk["page_end"]) for chunk in chunks] == [(1, 1), (2, 2)]
