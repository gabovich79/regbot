"""Additive v2 storage. Legacy chunks are never overwritten by the new index."""
import json
import uuid
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_versions (
 id TEXT PRIMARY KEY, document_id INTEGER NOT NULL, source_hash TEXT NOT NULL,
 embedding_model TEXT NOT NULL, card TEXT NOT NULL, issues TEXT NOT NULL,
 review_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS evidence_chunks (
 id TEXT PRIMARY KEY, version_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
 content TEXT NOT NULL, context TEXT NOT NULL, section TEXT NOT NULL,
 section_text TEXT NOT NULL, page_start INTEGER, page_end INTEGER,
 embedding TEXT NOT NULL, UNIQUE(version_id, ordinal)
);
CREATE INDEX IF NOT EXISTS evidence_chunks_version ON evidence_chunks(version_id);
CREATE TABLE IF NOT EXISTS index_releases (
 id TEXT PRIMARY KEY, manifest TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS active_index (singleton INTEGER PRIMARY KEY CHECK(singleton=1), release_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS request_traces (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, conversation_id INTEGER,
 status TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS demo_requests (
 id TEXT PRIMARY KEY, day TEXT NOT NULL, owner TEXT NOT NULL, ip_hash TEXT NOT NULL,
 reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL,
 expires_at TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS demo_requests_day ON demo_requests(day);
CREATE TABLE IF NOT EXISTS owned_conversations (conversation_id INTEGER PRIMARY KEY, owner TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS provider_costs (
 id INTEGER PRIMARY KEY, request_id TEXT, purpose TEXT NOT NULL, model TEXT NOT NULL,
 input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


async def migrate(db):
    await db.executescript(SCHEMA)


async def stage(db, document_id, source_hash, model, card, issues, chunks, vectors):
    if not chunks or len(chunks) != len(vectors):
        raise ValueError('Cannot stage an empty or incomplete index')
    version = uuid.uuid4().hex
    # All network operations have completed before the transaction starts.
    try:
        await db.execute('BEGIN IMMEDIATE')
        await db.execute('INSERT INTO evidence_versions(id,document_id,source_hash,embedding_model,card,issues) VALUES(?,?,?,?,?,?)',
                         (version, document_id, source_hash, model, json.dumps(card, ensure_ascii=False), json.dumps(issues)))
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            await db.execute('INSERT INTO evidence_chunks VALUES(?,?,?,?,?,?,?,?,?,?)',
                (f'D{document_id}-V{version}-C{i+1}', version, i, chunk['content'], chunk['context'],
                 chunk['section'], chunk['section_text'], chunk.get('page_start'), chunk.get('page_end'), json.dumps(list(vector))))
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    return version


async def review(db, version, accepted, note=''):
    row = await (await db.execute('SELECT issues,card FROM evidence_versions WHERE id=?', (version,))).fetchone()
    if not row:
        raise ValueError('Unknown index version')
    issues = json.loads(row['issues'])
    fatal = {'empty_extraction','invalid_characters','extraction_binary','extraction_hebrew_reversed',
             'extraction_embedded_foreign_glyphs',
             'empty_pages_require_visual_review_or_ocr','missing_original_requires_recovery', 'corrupt'}
    if accepted and await has_broken_glyphs(db,version):
        issues=list(dict.fromkeys(issues+['extraction_embedded_foreign_glyphs']))
    if accepted and (fatal.intersection(issues) or (issues and len(note.strip()) < 20)):
        raise ValueError('Extraction issues must be resolved by re-ingestion before approval')
    card = json.loads(row['card'])
    card['extraction_review'] = {'accepted':accepted,'note':note,'issues_acknowledged':issues,
                                'at':datetime.now(timezone.utc).isoformat()}
    await db.execute('UPDATE evidence_versions SET review_status=?,card=? WHERE id=?',
                     ('approved' if accepted else 'rejected',json.dumps(card,ensure_ascii=False), version))
    await db.commit()


async def has_broken_glyphs(db, version):
    # Recheck staged versions made before this detector existed. Old approval
    # must not bypass a newly measured extraction fault during activation.
    from services.document_integrity_service import embedded_foreign_glyphs
    rows=await (await db.execute('SELECT content FROM evidence_chunks WHERE version_id=? ORDER BY ordinal',(version,))).fetchall()
    return embedded_foreign_glyphs('\n'.join(r['content'] for r in rows))


async def activate(db, versions):
    """A release is a complete, explicitly reviewed manifest of active documents."""
    if not versions or len(set(versions)) != len(versions):
        raise ValueError('Supply a nonempty unique version manifest')
    try:
        await db.execute('BEGIN IMMEDIATE')
        active = {r['id'] for r in await (await db.execute('SELECT id FROM documents WHERE is_active=1')).fetchall()}
        selected = []
        models = set()
        for version in versions:
            row = await (await db.execute('SELECT * FROM evidence_versions WHERE id=?', (version,))).fetchone()
            if not row or row['review_status'] != 'approved':
                raise ValueError('Every version must have passed extraction review')
            if await has_broken_glyphs(db,version):
                raise ValueError('Extraction has embedded foreign glyphs; recover the source and re-index')
            selected.append(row['document_id'])
            models.add(row['embedding_model'])
        if set(selected) != active or len(selected) != len(active):
            raise ValueError('Manifest must contain exactly one version for every active document')
        if len(models) != 1:
            raise ValueError('Cannot mix embedding models')
        release = uuid.uuid4().hex
        await db.execute('INSERT INTO index_releases(id,manifest) VALUES(?,?)', (release, json.dumps(versions)))
        await db.execute('INSERT INTO active_index VALUES(1,?) ON CONFLICT(singleton) DO UPDATE SET release_id=excluded.release_id', (release,))
        for version, document_id in zip(versions, selected):
            await db.execute("UPDATE documents SET index_status='ready',index_error=NULL,chunk_count=(SELECT COUNT(*) FROM evidence_chunks WHERE version_id=?) WHERE id=?", (version,document_id))
        await db.commit()
        return release
    except BaseException:
        await db.rollback()
        raise


async def active_chunks(db):
    row = await (await db.execute('SELECT r.id,r.manifest FROM active_index a JOIN index_releases r ON a.release_id=r.id')).fetchone()
    if not row:
        return None, []
    versions = json.loads(row['manifest'])
    return row['id'], await version_chunks(db, versions, approved_only=True)


async def version_chunks(db, versions, *, approved_only=True):
    """Read cards once per version and share repeated parent text in memory.

    Do not materialize a SQL join which repeats a large document card and full
    parent section for every chunk before Python can deduplicate them.
    """
    result=[]
    for version in versions:
        metadata=await (await db.execute('SELECT v.* FROM evidence_versions v JOIN documents d ON d.id=v.document_id WHERE v.id=? AND d.is_active=1',(version,))).fetchone()
        if not metadata or (approved_only and metadata['review_status']!='approved'):continue
        shared={key:metadata[key] for key in ('document_id','card','embedding_model','source_hash')}
        parents={}
        async with db.execute('SELECT * FROM evidence_chunks WHERE version_id=? ORDER BY ordinal',(version,)) as cursor:
            async for source in cursor:
                chunk=dict(source)
                parent=chunk['section_text']
                chunk['section_text']=parents.setdefault(parent,parent)
                chunk.update(shared)
                result.append(chunk)
    return result


async def save_trace(db, request_id, owner, conversation_id, status, payload):
    await db.execute('INSERT OR REPLACE INTO request_traces(id,owner,conversation_id,status,payload) VALUES(?,?,?,?,?)',
        (request_id, owner, conversation_id, status, json.dumps(payload, ensure_ascii=False)))
    await db.commit()


async def prune(db):
    await db.execute("DELETE FROM request_traces WHERE created_at < datetime('now','-30 days')")
    await db.execute("DELETE FROM demo_requests WHERE created_at < datetime('now','-30 days')")
    # Only public demo conversations; legacy/admin history is not silently deleted.
    await db.execute("DELETE FROM messages WHERE conversation_id IN (SELECT conversation_id FROM owned_conversations) AND created_at < datetime('now','-30 days')")
    await db.execute('DELETE FROM conversations WHERE id IN (SELECT conversation_id FROM owned_conversations) AND NOT EXISTS (SELECT 1 FROM messages WHERE conversation_id=conversations.id)')
    await db.execute('DELETE FROM owned_conversations WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE id=conversation_id)')
    await db.commit()
