import anthropic
import time
import re
from config import ANTHROPIC_API_KEY, DEFAULT_MODEL, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, MAX_PROMPT_TOKENS, PRICING


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _estimate_tokens(text: str) -> int:
    """Quick token estimate: ~1.3 tokens per word for Hebrew/mixed text."""
    return int(len(text.split()) * 1.3)


def _build_documents_text(documents_texts: list[dict]) -> str:
    """Build concatenated documents text, truncating if total exceeds MAX_PROMPT_TOKENS."""
    # Reserve tokens for system prompt, user question, and output
    system_tokens = _estimate_tokens(SYSTEM_PROMPT)
    reserved = system_tokens + MAX_OUTPUT_TOKENS + 5000  # 5K buffer for question + history
    available_tokens = MAX_PROMPT_TOKENS - reserved

    parts = []
    used_tokens = 0
    skipped = []

    for doc in documents_texts:
        doc_text = f"=== מסמך: {doc['title']} ===\n{doc['text']}\n{'='*50}"
        doc_tokens = _estimate_tokens(doc_text)

        if used_tokens + doc_tokens > available_tokens:
            # Try to include a truncated version
            remaining_tokens = available_tokens - used_tokens
            if remaining_tokens > 2000:  # Worth including partial
                # Rough char estimate: ~4 chars per token for Hebrew
                max_chars = remaining_tokens * 4
                truncated_text = doc['text'][:max_chars]
                parts.append(
                    f"=== מסמך: {doc['title']} (קטוע — המסמך גדול מדי) ===\n"
                    f"{truncated_text}\n[... המסמך נחתך עקב מגבלת גודל ...]\n{'='*50}"
                )
                used_tokens += remaining_tokens
            else:
                skipped.append(doc['title'])
            break  # No room for more documents
        else:
            parts.append(doc_text)
            used_tokens += doc_tokens

    # Note skipped documents
    remaining_docs = documents_texts[len(parts) + (1 if skipped else 0):]
    for doc in remaining_docs:
        skipped.append(doc['title'])

    if skipped:
        parts.append(f"\n⚠️ המסמכים הבאים לא נכללו עקב מגבלת גודל: {', '.join(skipped)}")

    return "\n\n".join(parts)


def calculate_cost(usage) -> float:
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = 0
    cache_write = 0
    if hasattr(usage, "cache_read_input_tokens"):
        cache_read = usage.cache_read_input_tokens or 0
    if hasattr(usage, "cache_creation_input_tokens"):
        cache_write = usage.cache_creation_input_tokens or 0

    cost = (
        (input_tokens / 1_000_000) * PRICING["input"]
        + (output_tokens / 1_000_000) * PRICING["output"]
        + (cache_read / 1_000_000) * PRICING["cache_read"]
        + (cache_write / 1_000_000) * PRICING["cache_write"]
    )
    return round(cost, 6)


def extract_confidence(text: str) -> str | None:
    match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


async def stream_chat(user_question: str, documents_texts: list[dict],
                       conversation_history: list[dict] | None = None):
    """
    Stream a chat response. Yields dicts with type 'text', 'usage', or 'error'.
    """
    if not ANTHROPIC_API_KEY:
        yield {"type": "error", "text": "שגיאה: מפתח API של Anthropic לא הוגדר. הגדר ANTHROPIC_API_KEY."}
        return

    all_docs_text = _build_documents_text(documents_texts)
    history = conversation_history or []

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = []
    if all_docs_text:
        user_content.append({
            "type": "text",
            "text": f"להלן כל המסמכים הרגולטוריים:\n\n{all_docs_text}",
            "cache_control": {"type": "ephemeral"},
        })
    user_content.append({
        "type": "text",
        "text": user_question,
    })
    messages.append({"role": "user", "content": user_content})

    start_time = time.time()
    full_text = ""

    try:
        with client.messages.stream(
            model=DEFAULT_MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield {"type": "text", "text": text}

            response = stream.get_final_message()
            usage = response.usage
            elapsed_ms = int((time.time() - start_time) * 1000)

            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cost = calculate_cost(usage)
            confidence = extract_confidence(full_text)

            yield {
                "type": "usage",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read,
                "cache_write_tokens": cache_write,
                "cost_usd": cost,
                "response_time_ms": elapsed_ms,
                "confidence": confidence,
                "full_text": full_text,
            }

    except anthropic.APIError as e:
        yield {"type": "error", "text": f"שגיאה מ-Anthropic API: {str(e)}"}
    except Exception as e:
        yield {"type": "error", "text": f"שגיאה בלתי צפויה: {str(e)}"}
