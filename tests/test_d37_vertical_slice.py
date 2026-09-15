from services.document_ingestion_service import build_ingestion_receipt
from services.legal_parser import build_legal_tree



def test_parser_preserves_page_range_for_a_section_and_its_conditions():
    root = build_legal_tree(
        [
            {"text": "פרק ראשון: מס הכנסה", "page_start": 43, "page_end": 43},
            {"text": "סעיף 9(16א) קרן השתלמות", "page_start": 43, "page_end": 43},
            {"text": "סכומים שמשך עובד מחשבונו בקרן השתלמות.", "page_start": 43, "page_end": 43},
            {"text": "הפטור מותנה בשש שנים, ובחריגים בשלוש שנים.", "page_start": 44, "page_end": 44},
        ],
        {"id": 37, "title": "פקודת מס הכנסה"},
    )

    chapter = next(node for node in root["children"] if node["heading"] == "פרק ראשון: מס הכנסה")
    section = next(node for node in chapter["children"] if node["heading"] == "סעיף 9(16א) קרן השתלמות")

    assert section["page_start"] == 43
    assert section["page_end"] == 44
    assert "שש שנים" in section["raw_text"]


def test_ingestion_receipt_projects_page_ranges_to_d37_evidence_nodes():
    pages = [
        {"page_number": 43, "text": "פרק ראשון: מס הכנסה\nסעיף 9(16א) קרן השתלמות\nסכומים שמשך עובד מחשבונו בקרן השתלמות."},
        {"page_number": 44, "text": "הפטור מותנה בשש שנים, ובחריגים בשלוש שנים."},
    ]
    receipt = build_ingestion_receipt(
        {"id": 37, "title": "פקודת מס הכנסה", "source_type": "official", "source_ref": "source"},
        "\n".join(page["text"] for page in pages),
        original_path="/source/D37.pdf",
        source_checksum="d37-checksum",
        pages=pages,
    )

    node = next(item for item in receipt["nodes"] if item["section_label"] == "סעיף 9(16א) קרן השתלמות")

    assert node["page_start"] == 43
    assert node["page_end"] == 44


def test_page_extracted_wrapped_lines_remain_in_the_open_evidence_node():
    root = build_legal_tree(
        [
            {"text": "א) (א) סכומים שמשך עובד מחשבונו בקרן השתלמות, לרבות הפרשי הצמדה, וכן16(", "page_number": 43},
            {"text": "שנים ממועד6 ריבית ורווחים אחרים שמקורם בהפקדה המוטבת אם חלפו", "page_number": 43},
            {"text": "נפטר העובד, יהיו הזכאים לקבלת הסכומים כאמור רשאים למשכם מקרן", "page_number": 43},
            {"text": ";ההשתלמות בפטור ממס", "page_number": 43},
        ],
        {"id": 37, "title": "פקודת מס הכנסה"},
    )

    section = root["children"][0]

    assert "ריבית ורווחים אחרים" in section["raw_text"]
    assert "נפטר העובד" in section["raw_text"]
    assert "ההשתלמות בפטור ממס" in section["raw_text"]
