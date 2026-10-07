from scripts.decision_comparison import source_only, quote_checks


def test_reference_answers_cannot_leak_into_generation():
    item = {'id':'e1','content':'original','title':'doc','requirements':['secret gold'],
            'expected_answer':'secret gold','professional_approval':True}
    assert source_only(item) == {'id':'e1','content':'original','title':'doc'}


def test_invented_quote_and_unknown_citation_are_detected():
    evidence = [{'id':'e1','content':'the rule starts in September'}]
    answer = {'claims':[{'text':'rule','citations':[{'id':'e1','quote':'October'}]},
                        {'text':'rule','citations':[{'id':'e2','quote':'September'}]}]}
    assert quote_checks(answer,evidence) == ['claim:0:nonliteral_quote','claim:1:unknown_source']


def test_provenance_does_not_claim_semantic_entailment():
    evidence = [{'id':'e1','content':'no entitlement'}]
    answer = {'claims':[{'text':'entitled','citations':[{'id':'e1','quote':'no entitlement'}]}]}
    # This only checks existence; contradiction needs separate source review.
    assert quote_checks(answer,evidence) == []
