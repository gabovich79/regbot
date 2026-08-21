from services.claude_service import append_evidence_citation_contract


def test_custom_instructions_always_receive_evidence_citation_contract():
    instructions = append_evidence_citation_contract("ענה בעברית ובקצרה")

    assert "ענה בעברית ובקצרה" in instructions
    assert "[D<doc>-P<page>]" in instructions
    assert "[[SOURCE D...]]" in instructions
