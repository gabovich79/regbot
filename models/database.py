import aiosqlite
import os
from config import DB_PATH, DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db():
    db = await get_db()
    try:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                title       TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref  TEXT,
                text_path   TEXT NOT NULL,
                token_count INTEGER,
                is_active   INTEGER DEFAULT 1,
                added_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY,
                session_id      TEXT NOT NULL,
                started_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id                  INTEGER PRIMARY KEY,
                conversation_id     INTEGER REFERENCES conversations(id),
                role                TEXT NOT NULL,
                content             TEXT NOT NULL,
                confidence          TEXT,
                response_time_ms    INTEGER,
                input_tokens        INTEGER,
                output_tokens       INTEGER,
                cache_read_tokens   INTEGER,
                cache_write_tokens  INTEGER,
                cost_usd            REAL,
                created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db.commit()
    finally:
        await db.close()


# --- Document queries ---

async def get_all_documents(active_only=True):
    db = await get_db()
    try:
        query = "SELECT * FROM documents"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY added_at DESC"
        cursor = await db.execute(query)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def add_document(title, source_type, source_ref, text_path, token_count):
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO documents (title, source_type, source_ref, text_path, token_count)
               VALUES (?, ?, ?, ?, ?)""",
            (title, source_type, source_ref, text_path, token_count),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def delete_document(doc_id):
    db = await get_db()
    try:
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
    finally:
        await db.close()


async def get_document(doc_id):
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_total_tokens():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT COALESCE(SUM(token_count), 0) as total FROM documents WHERE is_active = 1"
        )
        row = await cursor.fetchone()
        return row["total"]
    finally:
        await db.close()


# --- Conversation queries ---

async def create_conversation(session_id):
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO conversations (session_id) VALUES (?)", (session_id,)
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_conversations():
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT c.*,
                   (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY id LIMIT 1) as first_question,
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
            FROM conversations c
            ORDER BY c.started_at DESC
        """)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def get_conversation_messages(conversation_id):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


# --- Message queries ---

async def save_message(conversation_id, role, content, confidence=None,
                       response_time_ms=None, input_tokens=None, output_tokens=None,
                       cache_read_tokens=None, cache_write_tokens=None, cost_usd=None):
    db = await get_db()
    try:
        cursor = await db.execute(
            """INSERT INTO messages
               (conversation_id, role, content, confidence, response_time_ms,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, role, content, confidence, response_time_ms,
             input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


# --- Logs queries ---

async def get_logs(page=1, per_page=20, date_from=None, date_to=None):
    db = await get_db()
    try:
        conditions = ["m.role = 'assistant'"]
        params = []
        if date_from:
            conditions.append("m.created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("m.created_at <= ?")
            params.append(date_to + " 23:59:59")

        where = " AND ".join(conditions)
        offset = (page - 1) * per_page

        count_cursor = await db.execute(
            f"SELECT COUNT(*) as total FROM messages m WHERE {where}", params
        )
        total = (await count_cursor.fetchone())["total"]

        cursor = await db.execute(
            f"""SELECT m.*,
                       (SELECT content FROM messages
                        WHERE conversation_id = m.conversation_id AND role = 'user'
                        ORDER BY id DESC LIMIT 1) as question
                FROM messages m
                WHERE {where}
                ORDER BY m.created_at DESC
                LIMIT ? OFFSET ?""",
            params + [per_page, offset],
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows], total
    finally:
        await db.close()


# --- Cost queries ---

async def get_costs_daily(days=7):
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT DATE(created_at) as date,
                      SUM(cost_usd) as total_cost,
                      COUNT(*) as query_count,
                      SUM(input_tokens) as total_input,
                      SUM(output_tokens) as total_output,
                      SUM(cache_read_tokens) as total_cache_read,
                      SUM(cache_write_tokens) as total_cache_write
               FROM messages
               WHERE role = 'assistant' AND created_at >= DATE('now', ?)
               GROUP BY DATE(created_at)
               ORDER BY date DESC""",
            (f"-{days} days",),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


async def get_costs_summary():
    db = await get_db()
    try:
        cursor = await db.execute(
            """SELECT
                COALESCE(SUM(CASE WHEN DATE(created_at) = DATE('now') THEN cost_usd ELSE 0 END), 0) as today,
                COALESCE(SUM(CASE WHEN created_at >= DATE('now', 'start of month') THEN cost_usd ELSE 0 END), 0) as this_month,
                COALESCE(SUM(cost_usd), 0) as total
               FROM messages WHERE role = 'assistant'"""
        )
        row = await cursor.fetchone()
        return dict(row)
    finally:
        await db.close()
