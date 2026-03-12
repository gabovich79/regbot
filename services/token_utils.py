"""
Accurate token estimation using tiktoken.

Hebrew text is extremely expensive in BPE tokenizers — each Hebrew character
can consume 2-4 tokens (vs ~0.25 tokens per English character). Using
character-length heuristics like `len(text)//3` massively underestimates
Hebrew token counts.

This module uses tiktoken's cl100k_base encoding (closest to Claude's tokenizer)
with a 20% safety margin to account for differences between OpenAI and Anthropic
tokenizers.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken with 20% safety margin.

    This provides accurate counts for Hebrew, English, and mixed text.
    The 20% margin accounts for differences between cl100k_base and
    Claude's actual tokenizer.
    """
    if not text:
        return 0
    count = len(_enc.encode(text))
    return int(count * 1.2)  # 20% safety margin
