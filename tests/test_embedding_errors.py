from services.embedding_errors import is_insufficient_quota


class _Error:
    def __init__(self, code=None, message=""):
        self.code = code
        self.message = message

    def __str__(self):
        return self.message


def test_detects_openai_insufficient_quota_by_error_code():
    assert is_insufficient_quota(_Error(code="insufficient_quota"))
    assert is_insufficient_quota(_Error(code="credit_balance_exhausted"))


def test_detects_openai_insufficient_quota_by_response_text():
    assert is_insufficient_quota(_Error(message="credit_balance_exhausted"))
    assert not is_insufficient_quota(_Error(code="rate_limit_exceeded"))
