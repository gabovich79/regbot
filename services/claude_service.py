import time
import re
import asyncio
import logging
import google.generativeai as genai
from google.generativeai.types import Tool as GeminiTool
from config import GOOGLE_API_KEY, DEFAULT_MODEL, SYSTEM_PROMPT, MAX_OUTPUT_TOKENS, PRICING
from services.rag_service import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

# Timeout for Gemini API calls (seconds)
GEMINI_TIMEOUT = 90


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


def _build_gemini_model(system_instructions: str):
    """Create a Gemini model instance with given instructions and Google Search grounding."""
    search_tool = GeminiTool(
        google_search_retrieval=genai.types.GoogleSearchRetrieval()
    )
    return genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        tools=[search_tool],
        system_instruction=system_instructions,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.3,
        ),
    )


def _sync_send_and_collect(chat, user_message: str) -> tuple[list[str], dict]:
    """
    Synchronous helper that sends message and collects all chunks.
    Runs in a thread pool to avoid blocking the event loop.
    Returns (text_chunks, usage_info).
    """
    chunks = []
    response = chat.send_message(user_message, stream=True)

    for chunk in response:
        if chunk.text:
            chunks.append(chunk.text)

    # Extract usage metadata
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

    return chunks, {"input_tokens": input_tokens, "output_tokens": output_tokens}


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
    yield {"type": "thinking", "text": "מעבד את השאלה..."}

    # Build Gemini model
    model = _build_gemini_model(system_instructions)

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
        # Run the blocking Gemini call in a thread pool with timeout
        # This prevents blocking the async event loop
        text_chunks, usage_meta = await asyncio.wait_for(
            asyncio.to_thread(_sync_send_and_collect, chat, user_message),
            timeout=GEMINI_TIMEOUT,
        )

        # Yield all text chunks
        for chunk_text in text_chunks:
            full_text += chunk_text
            yield {"type": "text", "text": chunk_text}

        # Calculate metrics
        elapsed_ms = int((time.time() - start_time) * 1000)
        cost = calculate_cost(usage_meta)
        confidence = extract_confidence(full_text)

        yield {
            "type": "usage",
            "input_tokens": usage_meta.get("input_tokens", 0),
            "output_tokens": usage_meta.get("output_tokens", 0),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": cost,
            "response_time_ms": elapsed_ms,
            "confidence": confidence,
            "full_text": full_text,
        }

    except asyncio.TimeoutError:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.warning(f"Gemini timeout after {elapsed_ms}ms for: {user_question[:50]}...")
        yield {"type": "error", "text": f"הזמן הקצוב לתשובה חלף ({GEMINI_TIMEOUT} שניות). נסה שוב."}

    except Exception as e:
        logger.error(f"Gemini error: {e}", exc_info=True)
        yield {"type": "error", "text": f"שגיאה מ-Gemini API: {str(e)}"}
