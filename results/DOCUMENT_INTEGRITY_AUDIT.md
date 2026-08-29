# Document Integrity Audit — 2026-08-29

**Status: production plan — findings from read-only dry-run, no production mutation.**

## Summary

- Documents audited: 36
- Verified: 31
- Warning: 5
- Failed: 0

The audit found 5 documents that need human attention. Two distinct problem
classes are now visible in the corpus that the flat retrieval layer never
surfaced.

## Findings

### 1. Document 25 — title/content mismatch (likely wrong file)

| Field | Value |
|---|---|
| Stored title | חוזר גמל 2019-9-14 גילוי עלות שנתית צפויה |
| Actual body title | אופן הפקדת תשלומים לקופת גמל - תיקון |
| Official number | 2019-9-14 |
| Evidence | "חוזר גופים מוסדיים 2019-9-14", "אופן הפקדת תשלומים לקופת גמל - תיקון" |
| Status | warning (title_body_mismatch) |

The stored title claims annual-cost disclosure; the actual circular is about
deposit of payments to provident funds. A question about annual cost could
retrieve this document through its title and answer from unrelated content.
Document 34 (annual-cost presentation) is the relevant source for that topic.

**Recommended action:** verify and fix the stored title/metadata for document
25, or replace the source with the actual 2019-9-14 cost-disclosure circular.

### 2. Documents 8 and 9 — reversed Hebrew extraction

Both PDFs return visually reversed Hebrew ("לכ :דוב ..." instead of "לכבוד:").
These are PDF text-extraction order bugs. Retrieval still works on tokens, but
any quote or snippet shown to a user is unreadable.

**Recommended action:** re-extract with an order-preserving pipeline (or OCR
fallback) before treating these documents as citable evidence.

### 3. Document 27 — binary/garbled extraction

Document 27 returns mostly binary with many null bytes (58 KB, ~18K nulls).
The stored `text_path` is not usable prose; citations to it are unreliable.

**Recommended action:** re-ingest from the original .doc source or OCR.

### 4. Documents 29 — minor mismatch

Document 29 title says periodic reports to members; body is about special
reports required from auditors. Likely a curator title choice, not a wrong
file, but worth confirming. `identity_evidence_missing` is expected because
the stored title is a curator label, not a title-page line.

## Design notes

- Profiles are generated from source text deterministically.
- Profile summaries and scope fields are retrieval metadata only; they are
  never cited as evidence.
- `identity_evidence` lines come verbatim from the title page / header block.
- Integrity statuses are advisory; they never mutate or archive documents.
- The dry-run did not write to any database.

## Generated artifacts

- `results/document_profile_audit.json` — full machine-readable audit.
- `eval/production_corpus_manifest_2026-08-29.json` — production manifest.
