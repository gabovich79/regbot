import sys
import os
import json
import uuid
import csv
import io
import logging

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config import DOCUMENTS_DIR, MAX_UPLOAD_SIZE_MB, MAX_TOKENS_WARNING, MAX_PROMPT_TOKENS, RAG_TOP_K, RAG_CONTEXT_WINDOW
from models.database import (
    init_db, get_all_documents, add_document, delete_document, get_document,
    get_total_tokens, create_conversation, get_conversations,
    get_conversation_messages, save_message, get_logs, get_costs_daily,
    get_costs_summary, get_db,
)
from services.document_service import (
    extract_pdf_bytes, extract_docx_bytes, fetch_url_text, fetch_gdrive_text,
    save_document_text, load_document_text, delete_document_file, estimate_tokens,
)
from services.claude_service import stream_chat
from services.rag_service import chunk_regulatory_document, embed_and_store_chunks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    yield


app = FastAPI(title="RegBot", lifespan=lifespan)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

BUILD_VERSION = "rag-embeddings-v1"


@app.get("/api/version")
async def get_version():
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM document_chunks")
        row = await cursor.fetchone()
        chunk_count = row["cnt"]
    except Exception:
        chunk_count = 0
    finally:
        await db.close()
    return {
        "version": BUILD_VERSION,
        "total_chunks": chunk_count,
    }


# --- Chat API ---

@app.post("/api/chat")
async def chat(
    question: str = Form(...),
    conversation_id: int = Form(None),
    session_id: str = Form(None),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    if not conversation_id:
        conversation_id = await create_conversation(session_id)

    # Save user message
    await save_message(conversation_id, "user", question)

    # Load conversation history
    history_rows = await get_conversation_messages(conversation_id)
    conversation_history = []
    for row in history_rows[:-1]:  # Exclude the just-saved user message
        conversation_history.append({"role": row["role"], "content": row["content"]})

    async def generate():
        usage_data = None
        db = await get_db()
        try:
            async for chunk in stream_chat(
                question, db,
                conversation_history,
                top_k=RAG_TOP_K,
                context_window=RAG_CONTEXT_WINDOW,
            ):
                if chunk["type"] == "text":
                    yield f"data: {json.dumps({'type': 'text', 'text': chunk['text']})}\n\n"
                elif chunk["type"] == "usage":
                    usage_data = chunk
                    yield f"data: {json.dumps({'type': 'usage', 'data': chunk})}\n\n"
                elif chunk["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'text': chunk['text']})}\n\n"
        finally:
            await db.close()

        if usage_data:
            await save_message(
                conversation_id, "assistant", usage_data["full_text"],
                confidence=usage_data.get("confidence"),
                response_time_ms=usage_data.get("response_time_ms"),
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                cache_read_tokens=usage_data.get("cache_read_tokens"),
                cache_write_tokens=usage_data.get("cache_write_tokens"),
                cost_usd=usage_data.get("cost_usd"),
            )

        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'session_id': session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Conversations API ---

@app.get("/api/conversations")
async def list_conversations():
    return await get_conversations()


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: int):
    messages = await get_conversation_messages(conv_id)
    if not messages:
        raise HTTPException(404, "שיחה לא נמצאה")
    return messages


# --- Documents API ---

@app.get("/api/documents")
async def list_documents():
    return await get_all_documents(active_only=False)


async def _index_document(doc_id: int, title: str, source_ref: str, text: str):
    """Chunk and embed a document for RAG retrieval."""
    db = await get_db()
    try:
        doc_metadata = {
            "id": doc_id,
            "title": title,
            "source_ref": source_ref,
            "effective_date": None,
            "topic": None,
        }
        chunks = chunk_regulatory_document(text, doc_metadata)
        num_chunks = await embed_and_store_chunks(chunks, db)
        logger.info(f"Document {doc_id} indexed: {num_chunks} chunks")
        return num_chunks
    finally:
        await db.close()


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"הקובץ גדול מ-{MAX_UPLOAD_SIZE_MB}MB")

    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        text = extract_pdf_bytes(content)
        source_type = "pdf"
    elif ext in ("doc", "docx"):
        text = extract_docx_bytes(content)
        source_type = "doc"
    else:
        raise HTTPException(400, "סוג קובץ לא נתמך. השתמש ב-PDF או DOC/DOCX")

    if not text.strip():
        raise HTTPException(400, "לא ניתן לחלץ טקסט מהקובץ")

    token_count = estimate_tokens(text)
    doc_id = await add_document(filename, source_type, filename, "", token_count)
    text_path = save_document_text(doc_id, text)

    db = await get_db()
    try:
        await db.execute("UPDATE documents SET text_path = ? WHERE id = ?", (text_path, doc_id))
        await db.commit()
    finally:
        await db.close()

    # RAG indexing
    try:
        num_chunks = await _index_document(doc_id, filename, filename, text)
    except Exception as e:
        logger.error(f"RAG indexing failed for doc {doc_id}: {e}")
        num_chunks = 0

    total = await get_total_tokens()
    warning = None
    if total > MAX_PROMPT_TOKENS:
        warning = (
            f"סה\"כ טוקנים ({total:,}) חורג ממגבלת ה-API ({MAX_PROMPT_TOKENS:,}). "
            f"המערכת תבחר אוטומטית את הקטעים הרלוונטיים ביותר לכל שאלה."
        )
    elif total > MAX_TOKENS_WARNING:
        warning = "המסמכים קרובים לגבול. המערכת תבחר אוטומטית את הקטעים הרלוונטיים ביותר."

    return {
        "id": doc_id,
        "title": filename,
        "token_count": token_count,
        "total_tokens": total,
        "num_chunks": num_chunks,
        "warning": warning,
        "message": f"נוסף בהצלחה — {token_count:,} טוקנים, {num_chunks} קטעים",
    }


@app.post("/api/documents/url")
async def add_document_url(url: str = Form(...), title: str = Form(None)):
    try:
        is_gdrive = "drive.google.com" in url or "docs.google.com" in url
        if is_gdrive:
            text = await fetch_gdrive_text(url)
            source_type = "gdrive"
        else:
            text = await fetch_url_text(url)
            source_type = "url"
    except Exception as e:
        raise HTTPException(400, f"שגיאה בהורדת המסמך: {str(e)}")

    if not text.strip():
        raise HTTPException(400, "לא ניתן לחלץ טקסט מהקישור")

    doc_title = title or url[:80]
    token_count = estimate_tokens(text)
    doc_id = await add_document(doc_title, source_type, url, "", token_count)
    text_path = save_document_text(doc_id, text)

    db = await get_db()
    try:
        await db.execute("UPDATE documents SET text_path = ? WHERE id = ?", (text_path, doc_id))
        await db.commit()
    finally:
        await db.close()

    # RAG indexing
    try:
        num_chunks = await _index_document(doc_id, doc_title, url, text)
    except Exception as e:
        logger.error(f"RAG indexing failed for doc {doc_id}: {e}")
        num_chunks = 0

    total = await get_total_tokens()
    warning = None
    if total > MAX_PROMPT_TOKENS:
        warning = (
            f"סה\"כ טוקנים ({total:,}) חורג ממגבלת ה-API ({MAX_PROMPT_TOKENS:,}). "
            f"המערכת תבחר אוטומטית את הקטעים הרלוונטיים ביותר לכל שאלה."
        )
    elif total > MAX_TOKENS_WARNING:
        warning = "המסמכים קרובים לגבול. המערכת תבחר אוטומטית את הקטעים הרלוונטיים ביותר."

    return {
        "id": doc_id,
        "title": doc_title,
        "token_count": token_count,
        "total_tokens": total,
        "num_chunks": num_chunks,
        "warning": warning,
        "message": f"נוסף בהצלחה — {token_count:,} טוקנים, {num_chunks} קטעים",
    }


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int):
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    delete_document_file(doc["text_path"])
    await delete_document(doc_id)
    return {"message": "המסמך הוסר בהצלחה"}


@app.get("/api/documents/stats")
async def document_stats():
    docs = await get_all_documents(active_only=True)
    total_tokens = sum(d.get("token_count", 0) or 0 for d in docs)
    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM document_chunks")
        row = await cursor.fetchone()
        chunk_count = row["cnt"]
    except Exception:
        chunk_count = 0
    finally:
        await db.close()
    return {
        "document_count": len(docs),
        "total_tokens": total_tokens,
        "total_chunks": chunk_count,
        "warning": total_tokens > MAX_TOKENS_WARNING,
    }


# --- Reindex API ---

@app.post("/api/documents/reindex")
async def reindex_all_documents():
    """Re-chunk and re-embed all active documents."""
    docs = await get_all_documents(active_only=True)
    results = []
    for doc in docs:
        try:
            text = load_document_text(doc["text_path"])
            num_chunks = await _index_document(
                doc["id"], doc["title"], doc.get("source_ref", ""), text
            )
            results.append({"id": doc["id"], "title": doc["title"], "chunks": num_chunks})
        except Exception as e:
            results.append({"id": doc["id"], "title": doc["title"], "error": str(e)})
    return {"results": results, "total_documents": len(docs)}


# --- Logs API ---

@app.get("/api/logs")
async def list_logs(
    page: int = Query(1, ge=1),
    date_from: str = Query(None),
    date_to: str = Query(None),
):
    logs, total = await get_logs(page, 20, date_from, date_to)
    return {"logs": logs, "total": total, "page": page, "per_page": 20}


@app.get("/api/logs/export")
async def export_logs():
    logs, _ = await get_logs(page=1, per_page=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["תאריך", "שאלה", "confidence", "זמן תגובה (ms)", "עלות ($)",
                     "input_tokens", "output_tokens", "cache_read", "cache_write"])
    for log in logs:
        writer.writerow([
            log.get("created_at", ""),
            (log.get("question", "") or "")[:100],
            log.get("confidence", ""),
            log.get("response_time_ms", ""),
            log.get("cost_usd", ""),
            log.get("input_tokens", ""),
            log.get("output_tokens", ""),
            log.get("cache_read_tokens", ""),
            log.get("cache_write_tokens", ""),
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=regbot_logs.csv"},
    )


# --- Costs API ---

@app.get("/api/costs")
async def get_costs():
    summary = await get_costs_summary()
    daily = await get_costs_daily(7)
    return {"summary": summary, "daily": daily}


# --- Static files (frontend) ---

app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/admin")
async def serve_admin():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))


@app.get("/logs")
async def serve_logs():
    return FileResponse(os.path.join(FRONTEND_DIR, "logs.html"))
