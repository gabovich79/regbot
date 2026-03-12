import anthropic
import time
import re
from config import ANTHROPIC_API_KEY, DEFAULT_MODEL, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, PRICING
from services.rag_service import retrieve_relevant_chunks


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


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


async def stream_chat(user_question: str, db,
                       conversation_history: list[dict] | None = None,
                       top_k: int = 20, context_window: int = 1):
    """
    Stream a chat response. Yields dicts with type 'text', 'usage', or 'error'.
    Uses embedding-based RAG to retrieve only the most relevant chunks.
    """
    if not ANTHROPIC_API_KEY:
        yield {"type": "error", "text": "שגיאה: מפתח API של Anthropic לא הוגדר. הגדר ANTHROPIC_API_KEY."}
        return

    # RAG: Retrieve relevant chunks via embedding similarity
    try:
        relevant_context = await retrieve_relevant_chunks(
            question=user_question,
            db=db,
            top_k=top_k,
            context_window=context_window,
        )
    except Exception as e:
        yield {"type": "error", "text": f"שגיאה בשליפת מסמכים: {str(e)}"}
        return

    history = conversation_history or []

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = []
    if relevant_context:
        user_content.append({
            "type": "text",
            "text": f"להלן קטעים רלוונטיים מהמסמכים הרגולטוריים:\n\n{relevant_context}",
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
