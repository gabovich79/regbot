import hashlib
import io

import pytest
from fastapi import HTTPException, UploadFile

import main


def upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


class FakeDb:
    async def execute(self, *_args, **_kwargs):
        return None

    async def commit(self):
        return None

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_upload_persists_validated_hierarchical_receipt_before_legacy_index(monkeypatch):
    content = b"verified-source-bytes"
    captured = {}

    async def add_document(*_args):
        return 77

    async def persist_receipt(_db, receipt, **_kwargs):
        captured["persisted_receipt"] = receipt
        return {"profile_records": 1, "node_records": 2, "fts_node_records": 2, "embedded_node_records": 0}

    async def index_document(*_args, **_kwargs):
        captured["legacy_index_called"] = True
        return 3

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "extract_docx_bytes", lambda _content: "מסמך רגולטורי\nסעיף 1\nטקסט מספק לצומת חוקי.")
    monkeypatch.setattr(main, "add_document", add_document)
    monkeypatch.setattr(main, "save_original_document", lambda _id, _ext, _content: ("/sources/77.docx", hashlib.sha256(content).hexdigest()))
    monkeypatch.setattr(main, "save_document_text", lambda _id, _text: "/texts/77.txt")
    monkeypatch.setattr(main, "update_document_source_artifact", noop)
    monkeypatch.setattr(main, "get_db", lambda: _return(FakeDb()))
    monkeypatch.setattr(main, "persist_ingestion_receipt", persist_receipt)
    monkeypatch.setattr(main, "_index_document", index_document)
    monkeypatch.setattr(main, "get_total_tokens", lambda: _return(100))
    monkeypatch.setattr(main, "build_ingestion_receipt", _validated_receipt)

    result = await main.upload_document(upload("regulation.docx", content))

    assert captured["persisted_receipt"]["source"]["checksum"] == hashlib.sha256(content).hexdigest()
    assert captured["persisted_receipt"]["document_id"] == 77
    assert captured["legacy_index_called"] is True
    assert result["hierarchical_ingestion"] == {"status": "validated", "node_records": 2}


@pytest.mark.asyncio
async def test_upload_rejects_legacy_doc_before_creating_a_document():
    with pytest.raises(HTTPException) as error:
        await main.upload_document(upload("legacy.doc", b"old word binary"))

    assert error.value.status_code == 400
    assert "DOCX" in error.value.detail


async def _return(value):
    return value


def _validated_receipt(document, _text, **kwargs):
    return {
        "document_id": document["id"],
        "status": "validated",
        "source": {"checksum": kwargs["source_checksum"]},
        "validation_errors": [],
    }
