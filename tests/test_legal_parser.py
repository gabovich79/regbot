from services.legal_parser import build_legal_tree


def test_parser_groups_docx_paragraphs_into_sections():
    paragraphs = [
        "י\"ג בתמוז התשע\"ו\n19 ביולי 2016",
        "חוזר גופים מוסדיים 2016-9-11\nסיווג: כללי",
        "העברת כספים בין קופות גמל - תיקון",
        "בתוקף סמכותי לפי סעיפים 23(ד) ו-39(ב) לחוק הפיקוח.",
        "כללי",
        "מטרת חוזר זה להסדיר את הליכי העברת הכספים בין קופות גמל.",
        "הגדרות",
        "\"החודש הקובע\" - החודש שבו חל המועד הקובע;",
        "טיפול בבקשת העברה",
        "הגוף המנהל של קופה מקבלת יבדוק אם ניתן לבצע העברה.",
    ]

    root = build_legal_tree(paragraphs, {"id": 1, "title": "חוזר"})

    nodes = [node for node in root["children"] if node["node_type"] in {"section", "chapter"}]
    assert "כללי" in [node["heading"] for node in nodes]
    assert "הגדרות" in [node["heading"] for node in nodes]
    assert "טיפול בבקשת העברה" in [node["heading"] for node in nodes]
    for node in nodes:
        assert node["raw_text"].strip()
        assert len(node["heading"]) < 80


def test_parser_keeps_nested_subsections_within_parent():
    paragraphs = [
        "פרק ראשון: המקור",
        "2. מקורות הכנסה",
        "2(א) הכנסה מעבודה",
        "2(ב) הכנסה מעסק",
        "פרק שני: המקום",
        "4. מקום ההכנסה",
    ]

    root = build_legal_tree(paragraphs, {"id": 1, "title": "חוק"})

    chapter = next(node for node in root["children"] if node["heading"] == "פרק ראשון: המקור")
    section = next(node for node in chapter["children"] if node["heading"] == "2. מקורות הכנסה")
    assert {child["heading"] for child in section["children"]} == {
        "2(א) הכנסה מעבודה",
        "2(ב) הכנסה מעסק",
    }
    assert any(
        child["heading"] == "4. מקום ההכנסה"
        for node in root["children"]
        if node["heading"] == "פרק שני: המקום"
        for child in node["children"]
    )
