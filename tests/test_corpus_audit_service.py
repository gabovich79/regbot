import sqlite3
from pathlib import Path

from services.corpus_audit_service import audit_corpus


def _create_corpus_db(path: Path, existing_text_path: Path):
    db = sqlite3.connect(path)
    try:
        db.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT,
                text_path TEXT NOT NULL,
                token_count INTEGER,
                is_active INTEGER DEFAULT 1,
                added_at DATETIME
            );
            CREATE TABLE document_chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                content TEXT NOT NULL
            );
            """
        )
        db.execute(
            """INSERT INTO documents
            (id, title, source_type, source_ref, text_path, token_count, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (1, "מסמך מאונדקס", "pdf", "https://example.test/a", str(existing_text_path), 12, 1),
        )
        db.execute(
            """INSERT INTO documents
            (id, title, source_type, source_ref, text_path, token_count, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (2, "מסמך ללא טקסט", "pdf", "https://example.test/b", str(path.parent / "missing.txt"), 0, 1),
        )
        db.execute(
            "INSERT INTO document_chunks (id, document_id, content) VALUES (?, ?, ?)",
            (1, 1, "קטע רגולטורי"),
        )
        db.commit()
    finally:
        db.close()


def test_audit_corpus_reports_index_and_extraction_problems(tmp_path):
    extracted = tmp_path / "1.txt"
    extracted.write_text("זהו טקסט שחולץ מקובץ PDF.", encoding="utf-8")
    db_path = tmp_path / "regbot.db"
    _create_corpus_db(db_path, extracted)

    report = audit_corpus(db_path)

    assert report["summary"] == {
        "documents": 2,
        "active_documents": 2,
        "indexed_documents": 1,
        "unindexed_documents": 1,
        "missing_text_files": 1,
        "total_chunks": 1,
    }
    assert report["documents"][0]["chunk_count"] == 1
    assert report["documents"][0]["extraction_chars"] == len(extracted.read_text(encoding="utf-8"))
    assert report["documents"][1]["issues"] == ["missing_text_file", "no_chunks"]
