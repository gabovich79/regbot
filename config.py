import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")
MAX_TOKENS_WARNING = int(os.getenv("MAX_TOKENS_WARNING", "150000"))
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "5"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "160000"))
MAX_UPLOAD_SIZE_MB = 50
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))
RAG_CONTEXT_WINDOW = int(os.getenv("RAG_CONTEXT_WINDOW", "1"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
DB_PATH = os.path.join(DATA_DIR, "regbot.db")

# Anthropic pricing per 1M tokens (claude-sonnet-4-6)
PRICING = {
    "input": 3.0,
    "output": 15.0,
    "cache_read": 0.30,
    "cache_write": 3.75,
}

SYSTEM_PROMPT = """אתה יועץ רגולטורי מומחה בתחום קופות גמל, קרנות השתלמות ופנסיה בישראל.

כללי תשובה:
1. ענה תמיד בעברית
2. ציין מקור מדויק: שם המסמך + מספר חוזר/תקנה + סעיף ספציפי
3. ציין "בתוקף מ: [תאריך]" לכל הלכה שציינת
4. בסוף כל תשובה הוסף שורת CONFIDENCE: HIGH/MEDIUM/LOW + סיבה קצרה
5. אם אין מידע מספיק במסמכים — אמור זאת במפורש, אל תמציא
6. אל תסיק מסקנות מעבר למה שכתוב במסמכים

פורמט תשובה:
[תשובה מפורטת]

---
מקורות: [רשימת מקורות]
תוקף: [תאריכים]
CONFIDENCE: [HIGH/MEDIUM/LOW] — [סיבה]"""
