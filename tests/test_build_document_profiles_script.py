import json

from scripts.build_document_profiles import build_profile_report


def test_build_profile_report_is_read_only_and_counts_integrity(tmp_path):
    manifest = [
        {
            "id": 22,
            "title": "חוזר גמל 2016-9-11 מסלולי השקעה תלויי גיל",
            "source_type": "url",
            "source_ref": "https://example.test/22.docx",
            "topic": None,
            "document_type": "חוזר",
            "effective_date": "2016-09-11",
            "valid_until": None,
            "lifecycle_status": "current",
        },
        {
            "id": 38,
            "title": "רשימת מסלולי השקעה – תיקון (2024-1471)",
            "source_type": "pdf",
            "source_ref": "source.pdf",
            "topic": "מסלולי השקעה",
            "document_type": "חוזר",
            "effective_date": None,
            "valid_until": None,
            "lifecycle_status": "current",
        },
    ]
    text_root = tmp_path / "texts"
    text_root.mkdir()
    (text_root / "22.txt").write_text(
        "חוזר גופים מוסדיים 2016-9-11\nהעברת כספים בין קופות גמל - תיקון\nמטרת החוזר להסדיר העברות.",
        encoding="utf-8",
    )
    (text_root / "38.txt").write_text(
        "רשימת מסלולי השקעה - תיקון\nמודל השקעות תלוי גיל\nמסלול לבני 50 ומטה, 50 עד 60, 60 ומעלה.",
        encoding="utf-8",
    )

    report = build_profile_report(manifest, text_root)

    assert report["count"] == 2
    assert report["integrity_counts"]["warning"] == 1
    assert report["documents"][0]["integrity"]["reasons"] == [
        "title_body_mismatch"
    ]
    assert not (tmp_path / "regbot.db").exists()
    json.dumps(report, ensure_ascii=False)
