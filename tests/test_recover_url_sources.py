import json

import pytest

from scripts import recover_url_sources


@pytest.mark.asyncio
async def test_url_recovery_dry_run_lists_only_url_backed_documents(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"id": 1, "title": "URL", "source_ref": "https://example.test/1.pdf"},
                {"id": 2, "title": "Filename", "source_ref": "2.pdf"},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(recover_url_sources, "MANIFEST", manifest)

    report = await recover_url_sources.recover(apply=False)

    assert report["summary"] == {"planned": 1}
    assert report["documents"] == [
        {
            "document_id": 1,
            "title": "URL",
            "url": "https://example.test/1.pdf",
            "status": "planned",
        }
    ]
