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


def _is_followup_question(question: str, history: list[dict]) -> bool:
    """Detect if a question is a short follow-up to an existing conversation."""
    if not history or len(history) < 2:
        return False
    # Short questions in an existing conversation are likely follow-ups
    words = question.split()
    if len(words) <= 12:
        return True
    # Explicit follow-up markers in Hebrew
    followup_markers = ["מה עם", "ומה לגבי", "תפרט", "תרחיב", "השורה התחתונה",
                        "בקיצור", "לסכם", "תסכם", "למעשה", "אז מה", "ואם",
                        "ובמקרה", "ולגבי", "איך זה", "למה"]
    q_lower = question.strip()
    for marker in followup_markers:
        if q_lower.startswith(marker):
            return True
    return False


async def get_system_instructions(db) -> str:
    """Load custom instructions from DB settings, fall back to config.py default."""
    try:
        cursor = await db.execute(
            "SELECT value FROM settings WHERE key = 'system_instructions'"
        )
        row = await cursor.fetchone()
        if row and row["value"]:
            return row["value"]
    except Exception:
        pass  # Table doesn't exist yet or other error
    return SYSTEM_PROMPT


async def stream_chat(user_question: str, db,
                       conversation_history: list[dict] | None = None,
                       top_k: int = 20, context_window: int = 1):
    """
    Stream a chat response using Google Gemini.
    Yields dicts with type 'text', 'thinking', 'usage', or 'error'.
    Uses embedding-based RAG to retrieve only the most relevant chunks.
    Smart follow-up detection: reduces RAG noise for clarification questions.
    """
    if not GOOGLE_API_KEY:
        yield {"type": "error", "text": "שגיאה: מפתח API של Google לא הוגדר. הגדר GOOGLE_API_KEY."}
        return

    history = conversation_history or []
    is_followup = _is_followup_question(user_question, history)

    # RAG: Retrieve relevant chunks (reduced for follow-ups)
    try:
        effective_top_k = 5 if is_followup else top_k
        relevant_context = await retrieve_relevant_chunks(
            question=user_question,
            db=db,
            top_k=effective_top_k,
            context_window=context_window,
        )
    except Exception as e:
        yield {"type": "error", "text": f"שגיאה בשליפת מסמכים: {str(e)}"}
        return

    # Load system instructions (from DB or config fallback)
    system_instructions = await get_system_instructions(db)

    # Yield thinking indicator so frontend knows we're working
    yield {"type": "thinking", "text": "מחפש מידע רלוונטי..."}

    # Build Gemini model with system instruction
    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        system_instruction=system_instructions,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.3,
        ),
    )

    # Convert conversation history to Gemini format
    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # Start chat with history
    chat = model.start_chat(history=gemini_history)

    # Build user message with RAG context
    if is_followup and relevant_context:
        user_message = (
            f"(הקשר נוסף מהמסמכים — השתמש רק אם רלוונטי לשאלת ההמשך):\n"
            f"{relevant_context}\n\n"
            f"שאלת המשך: {user_question}"
        )
    elif relevant_context:
        user_message = f"להלן קטעים רלוונטיים מהמסמכים הרגולטוריים:\n\n{relevant_context}\n\n{user_question}"
    else:
        user_message = user_question

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
