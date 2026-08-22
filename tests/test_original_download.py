import pytest

import main


@pytest.mark.asyncio
async def test_original_endpoint_returns_retained_source_file(tmp_path, monkeypatch):
    source = tmp_path / "42.pdf"
    source.write_bytes(b"%PDF-test")
    monkeypatch.setattr(
        main,
        "get_document",
        lambda _doc_id: None,
    )

    async def fake_get_document(_doc_id):
        return {"id": 42, "title": "מסמך מקור.pdf", "original_path": str(source)}

    monkeypatch.setattr(main, "get_document", fake_get_document)
    response = await main.download_original_document(42)

    assert response.path == str(source)
    assert response.filename == "מסמך מקור.pdf"


@pytest.mark.asyncio
async def test_original_endpoint_rejects_missing_source(monkeypatch):
    async def fake_get_document(_doc_id):
        return {"id": 42, "title": "מסמך", "original_path": None}

    monkeypatch.setattr(main, "get_document", fake_get_document)
    with pytest.raises(main.HTTPException) as error:
        await main.download_original_document(42)
    assert error.value.status_code == 404
