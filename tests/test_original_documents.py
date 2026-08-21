import hashlib

from services import document_service


def test_save_original_document_persists_bytes_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(document_service, "ORIGINALS_DIR", str(tmp_path))
    content = b"%PDF-original-regulatory-source"

    path, checksum = document_service.save_original_document(42, "pdf", content)

    assert path.endswith("42.pdf")
    assert open(path, "rb").read() == content
    assert checksum == hashlib.sha256(content).hexdigest()
