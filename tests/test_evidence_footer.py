from services.claude_service import append_retrieved_sources


def test_append_retrieved_sources_adds_compact_evidence_footer():
    context = """
[[SOURCE D22-C3 | מסמך: חוזר גמל 2016-9-11 מסלולי תלויי גיל | עמוד לא זמין במאגר הישן | סעיף: קטע 3 | URL: https://example.test/22]]
text
[[/SOURCE D22-C3]]
[[SOURCE D22-C4 | מסמך: חוזר גמל 2016-9-11 מסלולי תלויי גיל | עמוד לא זמין במאגר הישן | סעיף: קטע 4 | URL: https://example.test/22]]
text
[[/SOURCE D22-C4]]
"""

    answer = append_retrieved_sources("תשובה ללא citation", context)

    assert "מקורות שנשלפו" in answer
    assert "D22-C3" in answer
    assert "D22-C4" in answer
    assert "חוזר גמל 2016-9-11 מסלולי תלויי גיל" in answer


def test_append_retrieved_sources_does_not_duplicate_existing_footer():
    context = "[[SOURCE D23-P2 | מסמך: חוזר | עמוד 2 | סעיף: 3 | URL: https://x]]"
    answer = append_retrieved_sources("תשובה\n\nמקורות שנשלפו:\n* D23-P2", context)

    assert answer.count("מקורות שנשלפו") == 1
