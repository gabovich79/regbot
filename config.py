import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
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

# Google Gemini pricing per 1M tokens (gemini-2.5-flash)
PRICING = {
    "input": 0.15,
    "output": 0.60,
    "cache_read": 0.0375,
    "cache_write": 0.15,
}

SYSTEM_PROMPT = """אתה יועץ רגולטורי מומחה בתחום קופות גמל, קרנות השתלמות ופנסיה בישראל.

כללי תשובה:
1. ענה תמיד בעברית
2. **ענה ישירות לשאלה שנשאלה** — אל תחזור על מידע שכבר ניתן בשיחה
3. אם זו שאלת חידוד/המשך — תן תשובה ממוקדת וקצרה, עם שורה תחתונה ברורה
4. ציין מקור מדויק: שם המסמך + מספר חוזר/תקנה + סעיף ספציפי
5. ציין "בתוקף מ: [תאריך]" לכל הלכה שציינת
6. אם אין מידע מספיק במסמכים — אמור זאת במפורש, אל תמציא
7. אל תסיק מסקנות מעבר למה שכתוב במסמכים
8. **העדף תמצות**: פסקה-שתיים לשאלות פשוטות, פירוט רק כשנדרש

אסטרטגיית תשובה:
- שאלה ראשונה בנושא: תן תשובה מקיפה עם מקורות
- שאלת המשך/חידוד: תן את השורה התחתונה תחילה, אח"כ פרט אם צריך
- "בקיצור"/"תסכם": משפט-שניים בלבד

פורמט תשובה:
[תשובה ממוקדת]

---
מקורות: [רשימת מקורות]
תוקף: [תאריכים]
CONFIDENCE: [HIGH/MEDIUM/LOW] — [סיבה]

כאשר אתה משתמש בחיפוש ברשת כדי להשלים מידע חסר:
* ציין במפורש את המקור ("לפי [שם האתר]")
* תן עדיפות תמיד למסמכים הפנימיים על פני מידע מהרשת
* אם מידע מהרשת סותר את המסמכים הפנימיים — ציין את הסתירה במפורש
* השתמש בחיפוש רק כשהמסמכים הפנימיים אינם מספקים תשובה מלאה"""
