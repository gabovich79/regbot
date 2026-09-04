from services.document_profile_service import (
    _extract_canonical_title,
    _strip_file_extension,
)


def test_strip_file_extension_removes_trailing_upload_extension():
    assert _strip_file_extension("חוק הפיקוח על קופות גמל, התשסה-2005.docx") == (
        "חוק הפיקוח על קופות גמל, התשסה-2005"
    )
    assert _strip_file_extension("תיקון 15.pdf") == "תיקון 15"


def test_canonical_title_prefers_curated_title_over_pdf_header():
    # Doc 18 in production has a title ending with an upload extension, and the
    # recovered source PDF opens with a Knesset Sefer HaChukkim header line.
    stored = "חוק הפיקוח על שירותים פיננסיים (קופות גמל), התשסה–2005.docx"
    text = (
        "10.8.2005 ,, ה' באב התשס\"ה2024ספר החוקים889 *2005―חוק הפיקוח על שירותים "
        "פיננסיים )קופות גמל(, התשס\"ה  פרק א': הוראות כלליות -בחוק זה .1\n"
        "בתוקף סמכותי לפי סעיפים 23 ו-60 לחוק הפיקוח"
    )

    title = _extract_canonical_title(stored, text, None)

    assert title == "חוק הפיקוח על שירותים פיננסיים (קופות גמל), התשסה–2005"
