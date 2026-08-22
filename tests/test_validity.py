from services.validity import extract_date_from_title, document_validity_status


def test_extract_date_from_year_first_circular_title():
    assert extract_date_from_title("חוזר גמל 2024-9-8 שיעורי תמותה") == "2024-09-08"
    assert extract_date_from_title("חוזר גמל 2020-9-2 העברת כספים") == "2020-09-02"
    assert extract_date_from_title("חוזר גמל 2016-9-11 מסלולי תלויי גיל") == "2016-09-11"


def test_extract_date_handles_zero_padded_and_separators():
    assert extract_date_from_title("הוראה 2024-09-08") == "2024-09-08"
    assert extract_date_from_title("הוראה 2024.9.8") == "2024-09-08"


def test_extract_date_returns_none_without_date():
    assert extract_date_from_title("חוזר ללא תאריך") is None


def test_validity_status_current():
    assert document_validity_status({"superseded_by": None, "valid_until": None}) == "current"


def test_validity_status_superseded():
    assert document_validity_status({"superseded_by": 5, "valid_until": None}) == "superseded"


def test_validity_status_expired():
    status = document_validity_status(
        {"superseded_by": None, "valid_until": "2020-01-01"}, today="2024-01-01"
    )
    assert status == "expired"


def test_validity_status_future_valid_until_stays_current():
    status = document_validity_status(
        {"superseded_by": None, "valid_until": "2030-01-01"}, today="2024-01-01"
    )
    assert status == "current"
