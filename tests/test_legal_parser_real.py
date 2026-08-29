from pathlib import Path

from docx import Document

from services.legal_parser import build_legal_tree

REGULATION_DOCX = Path(
    "/Users/guygabovich/Downloads/regulation_h_2016-9-11.docx"
)


def test_parser_on_real_circular_docx():
    if not REGULATION_DOCX.exists():
        return
    document = Document(str(REGULATION_DOCX))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    root = build_legal_tree(paragraphs, {"id": 1, "title": "העברת כספים בין קופות גמל"})

    headings = [
        node["heading"]
        for node in root["children"]
        if node["node_type"] in {"chapter", "section"}
    ]
    assert "כללי" in headings
    assert "הגדרות" in headings
    assert any("טיפול בבקשת העברה" in heading for heading in headings)
    assert any(node["raw_text"].strip() for node in root["children"])


def test_parser_keeps_section_text_after_heading():
    paragraphs = [
        "כללי",
        "מטרת חוזר זה להסדיר את הליכי העברת הכספים בין קופות גמל.",
        "הגדרות",
        "\"החודש הקובע\" - החודש שבו חל המועד הקובע;",
    ]

    root = build_legal_tree(paragraphs, {"id": 1, "title": "חוזר"})

    general = next(node for node in root["children"] if node["heading"] == "כללי")
    definitions = next(node for node in root["children"] if node["heading"] == "הגדרות")
    assert "מטרת חוזר זה" in general["raw_text"]
    assert "החודש הקובע" in definitions["raw_text"]
