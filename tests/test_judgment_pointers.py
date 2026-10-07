import pytest

from scripts.diagnostic_judgment import judgment_schema, resolve_judgment


def test_reference_ids_cannot_be_used_as_retrieved_evidence():
    case = {'requirements':[{'id':'R1'}]}
    answer = {'text':'A supported statement'}
    evidence = [{'id':'original-document:42','content':'Original evidence'}]
    schema = judgment_schema(case, answer, evidence)
    fields = schema['properties']['requirements']['items']['properties']
    assert fields['evidence_ids']['items']['enum'] == ['E1']
    raw = {'requirements':[{'answer_span_ids':['A1'],'evidence_ids':['original-document:42']}]}
    with pytest.raises(ValueError, match='Unknown retrieved span'):
        resolve_judgment(raw, answer, evidence)
    raw['requirements'][0]['evidence_ids'] = ['E1']
    result = resolve_judgment(raw, answer, evidence)
    assert result['requirements'][0]['evidence_quotes'] == [
        {'source_id':'original-document:42','quote':'Original evidence'}]
