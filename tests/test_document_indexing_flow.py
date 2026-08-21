import pytest

import main


class _Db:
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_index_document_marks_document_ready_after_chunks_are_stored(monkeypatch):
    async def fake_get_db():
        return _Db()

    async def fake_embed(_chunks, _db):
        return 4

    status_updates = []

    async def capture_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda _text, _metadata: [{"content": "x"}])
    monkeypatch.setattr(main, "embed_and_store_chunks", fake_embed)
    monkeypatch.setattr(main, "update_document_index_status", capture_status, raising=False)

    chunks = await main._index_document(42, "מסמך", "ref", "תוכן")

    assert chunks == 4
    assert status_updates == [((42, "ready"), {"chunk_count": 4})]


@pytest.mark.asyncio
async def test_index_document_marks_document_failed_when_embedding_fails(monkeypatch):
    async def fake_get_db():
        return _Db()

    async def failing_embed(_chunks, _db):
        raise RuntimeError("embedding provider unavailable")

    status_updates = []

    async def capture_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda _text, _metadata: [{"content": "x"}])
    monkeypatch.setattr(main, "embed_and_store_chunks", failing_embed)
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

    async def fake_embed(chunks, _db):
        assert [(chunk["page_start"], chunk["page_end"]) for chunk in chunks] == [(4, 4)]
        return 1

    async def ignore_status(*_args, **_kwargs):
        pass

    monkeypatch.setattr(main, "get_db", fake_get_db)
    monkeypatch.setattr(main, "chunk_regulatory_pages", lambda pages, metadata: [{
        "content": pages[0]["text"],
        "page_start": metadata["page_start"] if "page_start" in metadata else pages[0]["page_number"],
        "page_end": pages[0]["page_number"],
    }], raising=False)
    monkeypatch.setattr(main, "chunk_regulatory_document", lambda *_args: (_ for _ in ()).throw(AssertionError("flat chunking used")))
    monkeypatch.setattr(main, "embed_and_store_chunks", fake_embed)
    monkeypatch.setattr(main, "update_document_index_status", ignore_status)

    result = await main._index_document(
        42, "מסמך", "ref", "טקסט שטוח", pages=[{"page_number": 4, "text": "טקסט בעמוד"}]
    )

    assert result == 1
