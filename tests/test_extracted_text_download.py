import pytest

import main


@pytest.mark.asyncio
async def test_extracted_text_endpoint_returns_saved_text(tmp_path, monkeypatch):
    source = tmp_path / "42.txt"
    source.write_text("תוכן מסמך", encoding="utf-8")

    async def fake_get_document(_doc_id):
        return {"id": 42, "title": "מסמך מקור.pdf", "text_path": str(source)}

    monkeypatch.setattr(main, "get_document", fake_get_document)
    response = await main.download_extracted_text(42)

    assert response.path == str(source)
    assert response.filename == "מסמך מקור.txt"


@pytest.mark.asyncio
async def test_extracted_text_endpoint_rejects_missing_text(monkeypatch):
    async def fake_get_document(_doc_id):
        return {"id": 42, "title": "מסמך", "text_path": None}

    monkeypatch.setattr(main, "get_document", fake_get_document)
    with pytest.raises(main.HTTPException) as error:
        await main.download_extracted_text(42)
    assert error.value.status_code == 404
