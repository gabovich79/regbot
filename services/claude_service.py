import anthropic
import time
import re
from config import ANTHROPIC_API_KEY, DEFAULT_MODEL, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, MAX_PROMPT_TOKENS, PRICING
from services.retrieval_service import retrieve_relevant_documents
from services.token_utils import estimate_tokens as _estimate_tokens


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _get_available_doc_tokens() -> int:
    """Calculate how many tokens are available for documents."""
    system_tokens = _estimate_tokens(SYSTEM_PROMPT)
    reserved = system_tokens + MAX_OUTPUT_TOKENS + 5000  # buffer for question + history
    return MAX_PROMPT_TOKENS - reserved


def _build_documents_text(documents_texts: list[dict], skipped_titles: list[str] = None) -> str:
    """Build concatenated documents text from pre-selected relevant documents."""
    parts = []
    for doc in documents_texts:
        parts.append(f"=== מסמך: {doc['title']} ===\n{doc['text']}\n{'='*50}")

    if skipped_titles:
        parts.append(
            f"\n⚠️ מסמכים נוספים קיימים במערכת אך לא נכללו בשל מגבלת גודל: "
            f"{', '.join(skipped_titles)}. "
            f"אם השאלה מתייחסת למסמכים אלו, בקש מהמשתמש לשאול שאלה ספציפית יותר."
        )

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
    Uses retrieval to select only the most relevant documents that fit the token budget.
    """
    if not ANTHROPIC_API_KEY:
        yield {"type": "error", "text": "שגיאה: מפתח API של Anthropic לא הוגדר. הגדר ANTHROPIC_API_KEY."}
        return

    # RAG: Select relevant documents that fit within token budget
    available_tokens = _get_available_doc_tokens()
    selected_docs, skipped_titles = retrieve_relevant_documents(
        question=user_question,
        documents=documents_texts,
        max_tokens=available_tokens,
    )

    all_docs_text = _build_documents_text(selected_docs, skipped_titles)
    history = conversation_history or []

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = []
    if all_docs_text:
        user_content.append({
            "type": "text",
            "text": f"להלן המסמכים הרגולטוריים הרלוונטיים ({len(selected_docs)} מתוך {len(documents_texts)} מסמכים):\n\n{all_docs_text}",
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
                "docs_used": len(selected_docs),
                "docs_total": len(documents_texts),
                "docs_skipped": skipped_titles,
            }

    except anthropic.APIError as e:
        yield {"type": "error", "text": f"שגיאה מ-Anthropic API: {str(e)}"}
    except Exception as e:
        yield {"type": "error", "text": f"שגיאה בלתי צפויה: {str(e)}"}
