import sys
import os
import json
import uuid
import csv
import io
import logging
import secrets
import asyncio
import time

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse, Response, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config import DOCUMENTS_DIR, MAX_UPLOAD_SIZE_MB, MAX_TOKENS_WARNING, MAX_PROMPT_TOKENS, RAG_TOP_K, RAG_CONTEXT_WINDOW, ADMIN_PASSWORD, SYSTEM_PROMPT
from models.database import (
    init_db, get_all_documents, add_document, delete_document, get_document,
    get_total_tokens, create_conversation, get_conversations,
    get_conversation_messages, save_message, get_logs, get_costs_daily,
    get_costs_summary, get_db, get_setting, set_setting,
    update_document_index_status, update_document_source_artifact,
    set_document_validity, update_document_metadata, archive_document,
)
from services.document_service import (
    extract_pdf_bytes, extract_pdf_bytes_pages, extract_docx_bytes, fetch_url_document, fetch_url_text, fetch_gdrive_text,
    save_original_document, save_document_text, load_document_text, delete_document_file, estimate_tokens,
)
from services.claude_service import stream_chat
from services.rag_service import chunk_regulatory_document, chunk_regulatory_pages, embed_and_store_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("RegBot starting up...")
        await init_db()
        from models.evidence_store import prune
        cleanup_db = await get_db()
        try:
            await prune(cleanup_db)
        finally:
            await cleanup_db.close()
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        logger.info("RegBot startup complete.")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise
    yield


app = FastAPI(title="RegBot", lifespan=lifespan)


@app.middleware('http')
async def origin_guard(request: Request, call_next):
    if request.method in {'POST','PUT','PATCH','DELETE'}:
        from services.public_access import same_origin
        try:
            same_origin(request)
        except HTTPException as exc:
            return JSONResponse({'detail':exc.detail}, status_code=exc.status_code)
    return await call_next(request)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

BUILD_VERSION = "evidence-pipeline-v2"

# --- Auth ---

security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_PASSWORD:
        if os.getenv("ALLOW_INSECURE_DEV") == "1":
            return
        raise HTTPException(503, "Admin authentication is not configured")
    correct = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not correct:
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials.username


@app.get("/api/version")
async def get_version():
    from config import DEFAULT_MODEL, EMBEDDING_MODEL
    db = await get_db()
    try:
        row = await (await db.execute('SELECT release_id FROM active_index WHERE singleton=1')).fetchone()
        count = await (await db.execute('SELECT COUNT(*) AS n FROM evidence_chunks')).fetchone()
        return {"version": BUILD_VERSION, "commit": os.getenv('RENDER_GIT_COMMIT', os.getenv('BUILD_COMMIT', 'unknown')),
                "active_index": row['release_id'] if row else None, "total_chunks": count['n'],
                "generation_model": DEFAULT_MODEL, "embedding_model": EMBEDDING_MODEL}
    finally:
        await db.close()


@app.post("/api/chat")
async def chat(request: Request, question: str = Form(...), conversation_id: int = Form(None), session_id: str = Form(None)):
    from services.public_access import identity, set_cookie, same_origin, client_ip, assert_owner, reserve, settle, RESERVATION
    from services.providers import Gateway
    from services.evidence_pipeline import run_pipeline
    from models.evidence_store import save_trace, prune
    same_origin(request)
    question = question.strip()
    if not question or len(question) > 4000:
        raise HTTPException(422, 'Question must contain 1–4000 characters')
    owner, cookie = identity(request)
    request_id = uuid.uuid4().hex
    db = await get_db()
    reserved = False
    try:
        await prune(db)
        if conversation_id:
            await assert_owner(db, conversation_id, owner)
        active = await (await db.execute('SELECT release_id FROM active_index WHERE singleton=1')).fetchone()
        if not active:
            raise HTTPException(503, 'האינדקס טרם נבדק והופעל.')
        await reserve(db, request_id, owner, client_ip(request))
        reserved = True
        if not conversation_id:
            conversation_id = await create_conversation('v2:' + owner)
            await db.execute('INSERT INTO owned_conversations VALUES(?,?)', (conversation_id, owner))
            await db.commit()
        await save_message(conversation_id, 'user', question)
        rows = await get_conversation_messages(conversation_id)
        history = [{'role':r['role'], 'content':r['content']} for r in rows[:-1]][-8:]
    except BaseException:
        if reserved:
            await settle(db, request_id, 0, 'failed_before_provider')
        raise
    finally:
        await db.close()

    async def generate():
        queue = asyncio.Queue()
        gateway = Gateway(request_id=request_id, limit=RESERVATION)
        trace = {'question':question, 'request_id':request_id}
        start = time.monotonic()
        async def progress(message):
            await queue.put({'type':'thinking', 'text':message, 'request_id':request_id})
        async def work():
            conn = await get_db()
            status = 'failed'
            try:
                result = await asyncio.wait_for(run_pipeline(question, history, conn, gateway, trace, progress), timeout=90)
                status = result['status']
                elapsed = int((time.monotonic()-start)*1000)
                await save_message(conversation_id, 'assistant', result['text'], confidence=status,
                                   response_time_ms=elapsed, cost_usd=gateway.spent)
                await queue.put({'type':'sources','sources':result['sources'], 'request_id':request_id})
                await queue.put({'type':'text', 'text':result['text']})
                await queue.put({'type':'usage','data':{'confidence':status,'response_time_ms':elapsed,'cost_usd':gateway.spent,'request_id':request_id}})
            except asyncio.TimeoutError:
                await queue.put({'type':'error','text':'הזמן הקצוב הסתיים ללא תשובה מאומתת. נסה למקד את השאלה.'})
                trace['error'] = 'deadline_exceeded'
            except asyncio.CancelledError:
                status = 'cancelled'
                trace['error'] = 'client_disconnected'
                raise
            except Exception as exc:
                logger.exception('Evidence pipeline failed: %s',request_id)
                trace['error'] = type(exc).__name__
                await queue.put({'type':'error','text':'לא ניתן להשלים תשובה מאומתת כעת. מזהה בדיקה: ' + request_id})
            finally:
                trace['provider_calls'] = gateway.calls
                trace['cost_usd'] = gateway.spent
                trace['response_time_ms'] = int((time.monotonic()-start)*1000)
                try:
                    await save_trace(conn,request_id,owner,conversation_id,status,trace)
                    await settle(conn,request_id,gateway.spent,status)
                finally:
                    await conn.close()
                    await queue.put(None)
        task = asyncio.create_task(work())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield 'data: ' + json.dumps(item,ensure_ascii=False) + '\n\n'
            await task
            yield 'data: ' + json.dumps({'type':'done','conversation_id':conversation_id,'request_id':request_id}) + '\n\n'
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    response = StreamingResponse(generate(), media_type='text/event-stream')
    set_cookie(response,cookie)
    response.headers['X-Request-ID'] = request_id
    return response


@app.get('/api/conversations')
async def list_conversations(request: Request):
    from services.public_access import identity, set_cookie
    owner,cookie = identity(request)
    db = await get_db()
    try:
        rows = await (await db.execute("""SELECT c.id,c.started_at,
            (SELECT content FROM messages WHERE conversation_id=c.id AND role='user' ORDER BY id LIMIT 1) AS first_question
            FROM conversations c JOIN owned_conversations o ON o.conversation_id=c.id
            WHERE o.owner=? ORDER BY c.id DESC""",(owner,))).fetchall()
        response = JSONResponse([dict(r) for r in rows])
        set_cookie(response,cookie)
        return response
    finally:
        await db.close()


@app.get('/api/conversations/{conv_id}')
async def get_conversation(conv_id: int, request: Request):
    from services.public_access import identity, assert_owner
    owner,_ = identity(request)
    db = await get_db()
    try:
        await assert_owner(db,conv_id,owner)
        return await get_conversation_messages(conv_id)
    finally:
        await db.close()


@app.get('/api/admin/conversations/{conv_id}')
async def admin_conversation(conv_id: int, _=Depends(verify_admin)):
    return await get_conversation_messages(conv_id)


@app.get('/api/admin/traces/{request_id}')
async def request_trace(request_id: str, _=Depends(verify_admin)):
    db = await get_db()
    try:
        row = await (await db.execute('SELECT * FROM request_traces WHERE id=?',(request_id,))).fetchone()
        if not row:
            raise HTTPException(404,'Trace not found')
        return {**dict(row),'payload':json.loads(row['payload'])}
    finally:
        await db.close()


@app.get('/api/admin/index/versions')
async def index_versions(_=Depends(verify_admin)):
    db = await get_db()
    try:
        rows = await (await db.execute('SELECT * FROM evidence_versions ORDER BY created_at DESC')).fetchall()
        return [{**dict(r),'card':{k:v for k,v in json.loads(r['card']).items() if k!='embedding'},'issues':json.loads(r['issues'])} for r in rows]
    finally:
        await db.close()


@app.get('/api/admin/index/versions/{version}')
async def inspect_index(version: str, _=Depends(verify_admin)):
    db = await get_db()
    try:
        row = await (await db.execute('SELECT * FROM evidence_versions WHERE id=?',(version,))).fetchone()
        if not row:
            raise HTTPException(404,'Version not found')
        chunks = await (await db.execute('SELECT id,ordinal,content,context,section,page_start,page_end FROM evidence_chunks WHERE version_id=? ORDER BY ordinal',(version,))).fetchall()
        return {'version':version,'card':{k:v for k,v in json.loads(row['card']).items() if k!='embedding'},
                'issues':json.loads(row['issues']),'review_status':row['review_status'],'chunks':[dict(c) for c in chunks]}
    finally:
        await db.close()


@app.post('/api/admin/index/versions/{version}/metadata')
async def verify_index_metadata(version: str, metadata: dict, _=Depends(verify_admin)):
    from datetime import date
    allowed = {'effective_date','valid_until','lifecycle_status','source_quote'}
    if set(metadata)-allowed or not isinstance(metadata.get('source_quote'),str) or len(metadata['source_quote']) < 10:
        raise HTTPException(422,'Supply exact supporting source_quote and only validity fields')
    for field in ('effective_date','valid_until'):
        if metadata.get(field):
            try:
                date.fromisoformat(metadata[field])
            except (TypeError,ValueError):
                raise HTTPException(422,'Invalid ISO date')
    if metadata.get('lifecycle_status') not in (None,'current','draft','superseded','expired','unknown'):
        raise HTTPException(422,'Unknown lifecycle status')
    db = await get_db()
    try:
        row = await (await db.execute('SELECT card,review_status FROM evidence_versions WHERE id=?',(version,))).fetchone()
        if not row or row['review_status'] != 'pending':
            raise HTTPException(422,'Only pending versions may be edited; re-index to change an approved version')
        texts = await (await db.execute('SELECT section_text FROM evidence_chunks WHERE version_id=?',(version,))).fetchall()
        if not any(metadata['source_quote'] in r['section_text'] for r in texts):
            raise HTTPException(422,'Supporting quote is not present in the source')
        card = json.loads(row['card'])
        card.update(metadata)
        card['metadata_verified'] = True
        await db.execute('UPDATE evidence_versions SET card=? WHERE id=?',(json.dumps(card,ensure_ascii=False),version))
        await db.commit()
        return {'version':version,'metadata_verified':True}
    finally:
        await db.close()


@app.get('/api/admin/runtime')
async def runtime_snapshot(_=Depends(verify_admin)):
    from config import DATA_DIR, DEFAULT_MODEL, EMBEDDING_MODEL
    from services.providers import prices
    from services.claude_service import get_system_instructions
    db = await get_db()
    try:
        return {'version':await get_version(), 'data_dir':os.path.abspath(DATA_DIR),
                'models':{'generation':DEFAULT_MODEL,'embeddings':EMBEDDING_MODEL},
                'price_registry':prices(), 'legacy_system_instructions':await get_system_instructions(db),
                'active_prompt_policy':'evidence_pipeline_v2 (legacy editable prompt is not used)',
                'public_sessions_configured':len(os.getenv('DEMO_SESSION_SECRET',''))>=32}
    finally:
        await db.close()


@app.post('/api/admin/index/{version}/review')
async def review_index(version: str, accepted: bool = Form(...), note: str = Form(''), _=Depends(verify_admin)):
    from models.evidence_store import review
    db = await get_db()
    try:
        await review(db,version,accepted,note)
        return {'version':version,'accepted':accepted}
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    finally:
        await db.close()


@app.post('/api/admin/index/activate')
async def activate_index(versions: list[str], _=Depends(verify_admin)):
    from models.evidence_store import activate
    db = await get_db()
    try:
        return {'release_id':await activate(db,versions)}
    except ValueError as exc:
        raise HTTPException(422,str(exc))
    finally:
        await db.close()


# --- Documents API ---

@app.get("/api/documents")
async def list_documents(_=Depends(verify_admin)):
    return await get_all_documents(active_only=False)


async def _index_document(
    doc_id: int,
    title: str,
    source_ref: str,
    text: str,
    pages: list[dict] | None = None,
):
    """Chunk and embed a document for RAG retrieval."""
    db = await get_db()
    try:
        from services.knowledge import stage_document
        stored = await get_document(doc_id)
        doc_metadata = {**(stored or {}), 'id':doc_id, 'title':title, 'source_ref':source_ref}
        original = doc_metadata.get('original_path')
        if pages is None and original and os.path.isfile(original):
            if original.lower().endswith('.pdf'):
                with open(original, 'rb') as handle:
                    pages = extract_pdf_bytes_pages(handle.read())
            elif original.lower().endswith('.docx'):
                with open(original, 'rb') as handle:
                    text = extract_docx_bytes(handle.read())
        version, num_chunks = await stage_document(db, doc_metadata, text, pages)
        logger.info('Staged document %s as version %s; activation requires review', doc_id, version)
        await update_document_index_status(doc_id, "staged", chunk_count=num_chunks)
        logger.info(f"Document {doc_id} indexed: {num_chunks} chunks")
        return num_chunks
    except Exception as error:
        try:
            await update_document_index_status(doc_id, "failed", error=str(error))
        except Exception:
            logger.exception("Failed to persist indexing failure for document %s", doc_id)
        raise
    finally:
        await db.close()


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), _=Depends(verify_admin)):
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"הקובץ גדול מ-{MAX_UPLOAD_SIZE_MB}MB")

    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    pages = None
    if ext == "pdf":
        pages = extract_pdf_bytes_pages(content)
        text = "\n\n".join(page["text"] for page in pages)
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
    original_path, source_checksum = save_original_document(doc_id, ext, content)
    await update_document_source_artifact(
        doc_id, original_path=original_path, checksum=source_checksum
    )
    text_path = save_document_text(doc_id, text)

    db = await get_db()
    try:
        await db.execute("UPDATE documents SET text_path = ? WHERE id = ?", (text_path, doc_id))
        await db.commit()
    finally:
        await db.close()

    # RAG indexing
    try:
        num_chunks = await _index_document(doc_id, filename, filename, text, pages=pages)
    except Exception as e:
        logger.error(f"RAG indexing failed for doc {doc_id}: {e}")
        num_chunks = 0

    total = await get_total_tokens()
    # Corpus size is not a prompt size: retrieval sends a bounded evidence
    # subset for every question. Health is shown through ready/failed status.
    warning = None

    return {
        "id": doc_id,
        "title": filename,
        "token_count": token_count,
        "total_tokens": total,
        "num_chunks": num_chunks,
        "index_status": "staged" if num_chunks else "failed",
        "warning": warning,
        "message": (
            f"נוסף והוכן לחיפוש — {token_count:,} טוקנים, {num_chunks} קטעים"
            if num_chunks else
            "הקובץ נשמר אך האינדוקס נכשל — פתח את טבלת המסמכים כדי לראות את סיבת הכשל"
        ),
    }


@app.post("/api/documents/url")
async def add_document_url(url: str = Form(...), title: str = Form(None), _=Depends(verify_admin)):
    pages = None
    source_content = None
    source_extension = None
    try:
        is_gdrive = "drive.google.com" in url or "docs.google.com" in url
        if is_gdrive:
            text = await fetch_gdrive_text(url)
            source_type = "gdrive"
        else:
            text, pages, source_content, source_extension = await fetch_url_document(url)
            source_type = "url"
    except Exception as e:
        raise HTTPException(400, f"שגיאה בהורדת המסמך: {str(e)}")

    if not text.strip():
        raise HTTPException(400, "לא ניתן לחלץ טקסט מהקישור")

    doc_title = title or url[:80]
    token_count = estimate_tokens(text)
    doc_id = await add_document(doc_title, source_type, url, "", token_count)
    if source_content is not None and source_extension is not None:
        original_path, source_checksum = save_original_document(
            doc_id, source_extension, source_content
        )
        await update_document_source_artifact(
            doc_id, original_path=original_path, checksum=source_checksum
        )
    text_path = save_document_text(doc_id, text)

    db = await get_db()
    try:
        await db.execute("UPDATE documents SET text_path = ? WHERE id = ?", (text_path, doc_id))
        await db.commit()
    finally:
        await db.close()

    # RAG indexing
    try:
        num_chunks = await _index_document(doc_id, doc_title, url, text, pages=pages)
    except Exception as e:
        logger.error(f"RAG indexing failed for doc {doc_id}: {e}")
        num_chunks = 0

    total = await get_total_tokens()
    warning = None

    return {
        "id": doc_id,
        "title": doc_title,
        "token_count": token_count,
        "total_tokens": total,
        "num_chunks": num_chunks,
        "index_status": "staged" if num_chunks else "failed",
        "warning": warning,
        "message": (
            f"נוסף והוכן לחיפוש — {token_count:,} טוקנים, {num_chunks} קטעים"
            if num_chunks else
            "המסמך נשמר אך האינדוקס נכשל — פתח את טבלת המסמכים כדי לראות את סיבת הכשל"
        ),
    }


@app.get("/api/documents/{doc_id}/original")
async def download_original_document(doc_id: int, _=Depends(verify_admin)):
    """Serve the retained original source artifact to an authenticated admin."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    original_path = doc.get("original_path")
    if not original_path or not os.path.isfile(original_path):
        raise HTTPException(404, "קובץ המקור המקורי לא נשמר עבור מסמך זה")
    filename = os.path.basename(doc.get("title") or original_path)
    return FileResponse(original_path, filename=filename, content_disposition_type="inline")


@app.get("/api/documents/{doc_id}/text")
async def download_extracted_text(doc_id: int, _=Depends(verify_admin)):
    """Serve the extracted UTF-8 text when the original binary is unavailable."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    text_path = doc.get("text_path")
    if not text_path or not os.path.isfile(text_path):
        raise HTTPException(404, "קובץ הטקסט שחולץ לא נשמר עבור מסמך זה")
    title = os.path.splitext(os.path.basename(doc.get("title") or "document"))[0]
    return FileResponse(
        text_path,
        media_type="text/plain; charset=utf-8",
        filename=f"{title}.txt",
        content_disposition_type="inline",
    )


@app.post("/api/documents/{doc_id}/archive")
async def archive_document_endpoint(doc_id: int, _=Depends(verify_admin)):
    """Archive a document from retrieval without deleting its source files."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    await archive_document(doc_id)
    return {"message": "המסמך הועבר לארכיון והוסר מהחיפוש"}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int, _=Depends(verify_admin)):
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    delete_document_file(doc["text_path"])
    await delete_document(doc_id)
    return {"message": "המסמך הוסר בהצלחה"}


@app.post("/api/documents/{doc_id}/validity")
async def set_document_validity_endpoint(
    doc_id: int,
    effective_date: str = Form(None),
    valid_until: str = Form(None),
    superseded_by: str = Form(None),
    _=Depends(verify_admin),
):
    """Set a document's effective date and supersession status."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")

    superseded_by_id: int | None = None
    if superseded_by not in (None, ""):
        superseded_by_id = int(superseded_by)
        if superseded_by_id == doc_id:
            raise HTTPException(400, "מסמך לא יכול להחליף את עצמו")
        replacement = await get_document(superseded_by_id)
        if not replacement:
            raise HTTPException(400, "המסמך המחליף לא נמצא")

    await set_document_validity(
        doc_id,
        effective_date=(effective_date or None),
        valid_until=(valid_until or None),
        superseded_by=superseded_by_id,
    )
    return {"message": "סטטוס התוקף עודכן בהצלחה"}


@app.post("/api/documents/{doc_id}/metadata")
async def update_document_metadata_endpoint(
    doc_id: int,
    title: str = Form(None),
    topic: str = Form(None),
    document_type: str = Form(None),
    lifecycle_status: str = Form(None),
    _=Depends(verify_admin),
):
    """Update curator metadata without re-indexing or deleting the document."""
    doc = await get_document(doc_id)
    if not doc:
        raise HTTPException(404, "מסמך לא נמצא")
    if title is not None and not title.strip():
        raise HTTPException(400, "כותרת לא יכולה להיות ריקה")
    try:
        await update_document_metadata(
            doc_id,
            title=title.strip() if title is not None else None,
            topic=topic.strip() if topic is not None else None,
            document_type=document_type.strip() if document_type is not None else None,
            lifecycle_status=lifecycle_status or None,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"message": "פרטי המסמך עודכנו בהצלחה"}


@app.get("/api/documents/stats")
async def document_stats(_=Depends(verify_admin)):
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
async def reindex_all_documents(_=Depends(verify_admin)):
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
    _=Depends(verify_admin),
):
    logs, total = await get_logs(page, 20, date_from, date_to)
    return {"logs": logs, "total": total, "page": page, "per_page": 20}


@app.get("/api/logs/export")
async def export_logs(_=Depends(verify_admin)):
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
async def get_costs(_=Depends(verify_admin)):
    summary = await get_costs_summary()
    daily = await get_costs_daily(7)
    return {"summary": summary, "daily": daily}


# --- Settings API ---

@app.get("/api/settings/instructions")
async def get_instructions(_=Depends(verify_admin)):
    custom = await get_setting("system_instructions")
    return {
        "instructions": custom or SYSTEM_PROMPT,
        "is_custom": custom is not None,
        "default": SYSTEM_PROMPT,
    }


@app.put("/api/settings/instructions")
async def update_instructions(
    instructions: str = Form(...),
    _=Depends(verify_admin),
):
    instructions = instructions.strip()
    if not instructions:
        raise HTTPException(400, "הוראות לא יכולות להיות ריקות")
    await set_setting("system_instructions", instructions)
    return {"message": "ההוראות עודכנו בהצלחה", "instructions": instructions}


@app.delete("/api/settings/instructions")
async def reset_instructions(_=Depends(verify_admin)):
    """Reset to default system prompt from config.py."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM settings WHERE key = 'system_instructions'")
        await db.commit()
    finally:
        await db.close()
    return {"message": "ההוראות אופסו לברירת מחדל", "instructions": SYSTEM_PROMPT}


# --- Static files (frontend) ---

app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/admin")
async def serve_admin(_=Depends(verify_admin)):
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))


@app.get("/logs")
async def serve_logs(_=Depends(verify_admin)):
    return FileResponse(os.path.join(FRONTEND_DIR, "logs.html"))
