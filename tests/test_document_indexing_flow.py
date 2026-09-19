import pytest

import main
from services import knowledge

@pytest.fixture(autouse=True)
def stored_document(monkeypatch):
    async def get_document(doc_id):
        return {"id":doc_id,"title":"מסמך","source_ref":"ref"}
    monkeypatch.setattr(main,"get_document",get_document)



class _Db:
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_index_document_marks_document_staged_without_activation(monkeypatch):
    async def fake_get_db():
        return _Db()

    async def fake_embed(_db, _metadata, _text, _pages=None):
        return "v1", 4

    status_updates = []

    async def capture_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda _text, _metadata: [{"content": "x"}])
    monkeypatch.setattr(knowledge, "stage_document", fake_embed)
    monkeypatch.setattr(main, "update_document_index_status", capture_status, raising=False)

    chunks = await main._index_document(42, "מסמך", "ref", "תוכן")

    assert chunks == 4
    assert status_updates == [((42, "staged"), {"chunk_count": 4})]


@pytest.mark.asyncio
async def test_index_document_marks_document_failed_when_embedding_fails(monkeypatch):
    async def fake_get_db():
        return _Db()

    async def failing_embed(_db, _metadata, _text, _pages=None):
        raise RuntimeError("embedding provider unavailable")

    status_updates = []

    async def capture_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda _text, _metadata: [{"content": "x"}])
    monkeypatch.setattr(knowledge, "stage_document", failing_embed)
    monkeypatch.setattr(main, "update_document_index_status", capture_status)

    with pytest.raises(RuntimeError, match="embedding provider unavailable"):
        await main._index_document(42, "מסמך", "ref", "תוכן")

    assert status_updates == [
        ((42, "failed"), {"error": "embedding provider unavailable"})
    ]


@pytest.mark.asyncio
async def test_index_document_uses_page_aware_chunks_when_pages_are_available(monkeypatch):
    async def fake_get_db():
        return _Db()

    async def fake_embed(_db, metadata, text, pages=None):
        assert pages == [{"page_number":4,"text":"טקסט בעמוד"}]
        return "v1", 1

    async def ignore_status(*_args, **_kwargs):
        pass

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_pages", lambda pages, metadata: [{
        "content": pages[0]["text"],
        "page_start": metadata["page_start"] if "page_start" in metadata else pages[0]["page_number"],
        "page_end": pages[0]["page_number"],
    }], raising=False)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda *_args: (_ for _ in ()).throw(AssertionError("flat chunking used")))
    monkeypatch.setattr(knowledge, "stage_document", fake_embed)
    monkeypatch.setattr(main, "update_document_index_status", ignore_status)

    result = await main._index_document(
        42, "מסמך", "ref", "טקסט שטוח", pages=[{"page_number": 4, "text": "טקסט בעמוד"}]
    )

    assert result == 1
