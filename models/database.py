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
                index_status TEXT NOT NULL DEFAULT 'pending',
                index_error  TEXT,
                indexed_at   DATETIME,
                chunk_count  INTEGER NOT NULL DEFAULT 0,
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

            CREATE TABLE IF NOT EXISTS document_chunks (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content        TEXT NOT NULL,
                section_header TEXT,
                chunk_index    INTEGER NOT NULL,
                document_title TEXT,
                document_ref   TEXT,
                effective_date TEXT,
                topic          TEXT,
                page_start     INTEGER,
                page_end       INTEGER,
                embedding      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_active ON document_chunks(document_id, chunk_index);

            CREATE TABLE IF NOT EXISTS settings (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await _migrate_document_indexing_columns(db)
        await _migrate_chunk_citation_columns(db)
        await db.commit()
    finally:
        await db.close()


async def _migrate_document_indexing_columns(db: aiosqlite.Connection):
    """Apply additive indexing-state migrations to existing SQLite databases."""
    cursor = await db.execute("PRAGMA table_info(documents)")
    columns = {row["name"] for row in await cursor.fetchall()}
    migrations = {
        "index_status": "TEXT NOT NULL DEFAULT 'pending'",
        "index_error": "TEXT",
        "indexed_at": "DATETIME",
        "chunk_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, definition in migrations.items():
        if column not in columns:
            await db.execute(f"ALTER TABLE documents ADD COLUMN {column} {definition}")

    # Backfill what can be known safely from the pre-existing chunks. Documents
    # without chunks are intentionally visible as failed instead of pretending
    # to be searchable.
    await db.execute("""
        UPDATE documents
        SET chunk_count = (
            SELECT COUNT(*) FROM document_chunks dc WHERE dc.document_id = documents.id
        )
    """)
    await db.execute("""
        UPDATE documents
        SET index_status = CASE WHEN chunk_count > 0 THEN 'ready' ELSE 'failed' END,
            index_error = CASE
                WHEN chunk_count = 0 THEN COALESCE(
                    index_error,
                    'לא נוצרו קטעי אינדקס; יש לבצע אינדוקס מחדש לאחר אימות ספק ה-embeddings.'
                )
                ELSE index_error
            END
        WHERE index_status = 'pending'
    """)


async def _migrate_chunk_citation_columns(db: aiosqlite.Connection):
    """Add nullable page boundaries for legacy chunks and future citations."""
    cursor = await db.execute("PRAGMA table_info(document_chunks)")
    columns = {row["name"] for row in await cursor.fetchall()}
    for column in ("page_start", "page_end"):
        if column not in columns:
            await db.execute(f"ALTER TABLE document_chunks ADD COLUMN {column} INTEGER")


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
            """INSERT INTO documents
               (title, source_type, source_ref, text_path, token_count, index_status)
               VALUES (?, ?, ?, ?, ?, 'indexing')""",
            (title, source_type, source_ref, text_path, token_count),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def update_document_index_status(
    doc_id: int,
    status: str,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
):
    """Persist a document's indexing lifecycle without hiding a failed state."""
    if status not in {"indexing", "ready", "failed"}:
        raise ValueError(f"Unsupported index status: {status}")

    db = await get_db()
    try:
        await db.execute(
            """
            UPDATE documents
            SET index_status = ?,
                index_error = ?,
                chunk_count = COALESCE(?, chunk_count),
                indexed_at = CASE WHEN ? = 'ready' THEN CURRENT_TIMESTAMP ELSE indexed_at END
            WHERE id = ?
            """,
            (status, error, chunk_count, status, doc_id),
        )
        await db.commit()
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


# --- Settings queries ---

async def get_setting(key: str) -> str | None:
    db = await get_db()
    try:
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        return row["value"] if row else None
    finally:
        await db.close()


async def set_setting(key: str, value: str):
    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP""",
            (key, value, value),
        )
        await db.commit()
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
