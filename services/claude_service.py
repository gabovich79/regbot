import time
import re
import google.generativeai as genai
from config import GOOGLE_API_KEY, DEFAULT_MODEL, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, PRICING
from services.rag_service import retrieve_relevant_chunks


genai.configure(api_key=GOOGLE_API_KEY)


def calculate_cost(usage) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (
        (input_tokens / 1_000_000) * PRICING["input"]
        + (output_tokens / 1_000_000) * PRICING["output"]
    )
    return round(cost, 6)


def extract_confidence(text: str) -> str | None:
    match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


async def stream_chat(user_question: str, db,
                       conversation_history: list[dict] | None = None,
                       top_k: int = 20, context_window: int = 1):
    """
    Stream a chat response using Google Gemini.
    Yields dicts with type 'text', 'usage', or 'error'.
    Uses embedding-based RAG to retrieve only the most relevant chunks.
    """
    if not GOOGLE_API_KEY:
        yield {"type": "error", "text": "שגיאה: מפתח API של Google לא הוגדר. הגדר GOOGLE_API_KEY."}
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

    # Build Gemini model with system instruction
    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.3,
        ),
    )

    # Convert conversation history to Gemini format
    gemini_history = []
    history = conversation_history or []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # Start chat with history
    chat = model.start_chat(history=gemini_history)

    # Build user message with RAG context
    user_message = ""
    if relevant_context:
        user_message += f"להלן קטעים רלוונטיים מהמסמכים הרגולטוריים:\n\n{relevant_context}\n\n"
    user_message += user_question

    start_time = time.time()
    full_text = ""

    try:
        response = chat.send_message(user_message, stream=True)

        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                yield {"type": "text", "text": chunk.text}

        # Get usage metadata
        elapsed_ms = int((time.time() - start_time) * 1000)

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        cost = calculate_cost(usage)
        confidence = extract_confidence(full_text)

        yield {
            "type": "usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": cost,
            "response_time_ms": elapsed_ms,
            "confidence": confidence,
            "full_text": full_text,
        }

    except Exception as e:
        yield {"type": "error", "text": f"שגיאה מ-Gemini API: {str(e)}"}
