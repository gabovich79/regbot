from services.document_service import normalize_source_url


def test_normalize_source_url_extracts_url_from_ui_wrapper():
    source_ref = "@url:`https://www.gov.il/BlobFolder/example.pdf`"
    assert normalize_source_url(source_ref) == "https://www.gov.il/BlobFolder/example.pdf"


def test_normalize_source_url_rejects_non_url_text():
    assert normalize_source_url("2021-9-5.pdf") is None
