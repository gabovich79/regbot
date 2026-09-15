from services.d37_evidence_answer_service import resolve_citations


def test_resolver_attaches_the_exact_verbatim_source_span_to_each_cited_id():
    answer = "עובד רשאי למשוך בפטור ממס [[D37-9-16A-A]]."
    evidence = [
        {
            "evidence_id": "D37-9-16A-A",
            "section": "9(16א)(א)",
            "page_start": 43,
            "page_end": 43,
            "raw_text": "סכומים שמשך עובד מחשבונו בקרן השתלמות",
        }
    ]

    resolved = resolve_citations(answer, evidence)

    assert resolved["answer"] == answer
    assert len(resolved["citations"]) == 1
    citation = resolved["citations"][0]
    assert citation["evidence_id"] == "D37-9-16A-A"
    assert citation["section"] == "9(16א)(א)"
    assert citation["page_start"] == 43
    assert citation["verbatim"] == "סכומים שמשך עובד מחשבונו בקרן השתלמות"


def test_resolver_marks_an_unknown_citation_id_instead_of_attaching_text():
    answer = "סעיף קובע [[D37-UNKNOWN]]."
    evidence = []

    resolved = resolve_citations(answer, evidence)

    assert resolved["citations"] == []
    assert resolved["unresolved_ids"] == ["D37-UNKNOWN"]
