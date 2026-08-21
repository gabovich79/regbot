from services.rag_service import fit_context_blocks_to_budget


def test_context_budget_keeps_blocks_within_token_limit():
    blocks = ["alpha " * 30, "beta " * 30, "gamma " * 30]

    selected = fit_context_blocks_to_budget(blocks, max_tokens=70)

    assert selected
    assert sum(len(block.split()) for block in selected) <= 70
    assert len(selected) < len(blocks)
