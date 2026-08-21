"""Classification helpers for embedding-provider errors."""


def is_insufficient_quota(error: Exception) -> bool:
    """Return true only for errors that cannot succeed by retrying."""
    code = getattr(error, "code", None)
    if code in {"insufficient_quota", "credit_balance_exhausted"}:
        return True
    return "credit_balance_exhausted" in str(error)
