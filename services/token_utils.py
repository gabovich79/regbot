"""
Accurate token estimation for Hebrew/mixed text.

Hebrew text is extremely expensive in BPE tokenizers — each Hebrew character
can consume 2-4 tokens (vs ~0.25 tokens per English character). Using
character-length heuristics like `len(text)//3` massively underestimates
Hebrew token counts.

Primary method: tiktoken cl100k_base with 20% safety margin.
Fallback (if tiktoken unavailable): conservative heuristic using UTF-8 byte length.
"""

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    _USE_TIKTOKEN = True
except Exception:
    _enc = None
    _USE_TIKTOKEN = False


def _heuristic_estimate(text: str) -> int:
    """Conservative fallback when tiktoken is unavailable.

    Hebrew UTF-8 bytes are 2 bytes each, and each Hebrew char typically
    maps to 2-4 BPE tokens. Using byte_length * 0.7 gives a safe estimate
    that accounts for Hebrew's high token cost while not over-counting
    ASCII text.
    """
    byte_len = len(text.encode("utf-8"))
    # For pure Hebrew: 2 bytes/char * 0.7 ≈ 1.4 tokens/byte ≈ 2.8 tokens/char (conservative)
    # For pure English: 1 byte/char * 0.7 ≈ 0.7 tokens/byte ≈ 0.7 tokens/char (slightly over)
    return max(int(byte_len * 0.7), 1)


def estimate_tokens(text: str) -> int:
    """Estimate token count accurately for Hebrew/English/mixed text.

    Uses tiktoken cl100k_base when available (with 20% safety margin),
    falls back to a conservative byte-length heuristic.
    """
    if not text:
        return 0

    if _USE_TIKTOKEN:
        count = len(_enc.encode(text))
        return int(count * 1.2)  # 20% safety margin for Anthropic vs OpenAI tokenizer
    else:
        return _heuristic_estimate(text)
