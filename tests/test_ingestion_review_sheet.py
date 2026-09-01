from scripts.build_ingestion_review_sheet import (
    classify_source_availability,
    source_text_for_review,
)


def test_classify_source_prefers_local_original_over_manifest_reference(tmp_path):
    original = tmp_path / "22.docx"
    original.write_bytes(b"original")
    document = {
        "id": 22,
        "source_ref": "22.docx",
        "original_path": "/var/data/originals/22.docx",
        "source_checksum": "a" * 64,
    }

    result = classify_source_availability(document, [tmp_path])

    assert result == {"status": "local_original", "path": str(original)}


def test_classify_source_marks_official_url_as_fetchable(tmp_path):
    document = {
        "id": 35,
        "source_ref": "https://www.gov.il/example.docx",
        "original_path": None,
        "source_checksum": None,
    }

    result = classify_source_availability(document, [tmp_path])

    assert result == {
        "status": "fetchable_url",
        "url": "https://www.gov.il/example.docx",
    }


def test_classify_source_detects_recovered_artifact_by_document_id(tmp_path):
    recovered = tmp_path / "35.docx"
    recovered.write_bytes(b"original")
    document = {
        "id": 35,
        "source_ref": "https://www.gov.il/example.docx",
        "original_path": None,
        "source_checksum": None,
    }

    result = classify_source_availability(document, [tmp_path])

    assert result == {"status": "local_original", "path": str(recovered)}


def test_review_uses_recovered_extracted_text_before_old_export(tmp_path):
    artifact = tmp_path / "35.docx"
    artifact.write_bytes(b"original")
    recovered_text = tmp_path / "35.txt"
    recovered_text.write_text("טקסט מקור משוחזר", encoding="utf-8")
    fallback = tmp_path / "fallback.txt"
    fallback.write_text("טקסט ישן", encoding="utf-8")

    text, origin = source_text_for_review(
        {"id": 35},
        {"status": "local_original", "path": str(artifact)},
        fallback,
    )

    assert text == "טקסט מקור משוחזר"
    assert origin == "recovered_source_text"


def test_classify_source_requires_reupload_without_url_or_artifact(tmp_path):
    document = {
        "id": 19,
        "source_ref": "2016-9-8-fees.pdf",
        "original_path": None,
        "source_checksum": None,
    }

    result = classify_source_availability(document, [tmp_path])

    assert result == {"status": "needs_reupload"}
