# Extractor & Parser Comparison — Task 3

**Status: production plan — findings from local spikes, no production mutation.**

## 1. Extraction measurements (current pipeline)

| Document | Kind | Size | Hebrew chars | Time |
|---|---|---|---|---|
| regulation_h_2016-9-11.docx | DOCX | 298 paragraphs / 32K chars | 24K | 0.08s |
| age-3.pdf (2024-1471) | PDF | 18 pages / 28K chars | 20K | 0.04s |
| Income Tax Ordinance (312p) | PDF | 312 pages / 801K chars | 553K | 1.10s |

Current extraction (PyMuPDF + python-docx) is fast and text-preserving for
normal PDFs and DOCX files.

## 2. Known extraction failures (found by integrity audit)

- Documents 8 & 9: Hebrew extracted visually reversed (RTL order bug).
- Document 27: binary/garbled extraction (many NUL bytes).
- Document 25: stored title describes annual-cost disclosure, body is a
  payments-deposit circular.

These are corpus issues the new integrity layer flags before they reach
retrieval.

## 3. Parser comparison

| Document | Old flat section matches | New hierarchical headings |
|---|---|---|
| regulation_h_2016-9-11.docx | 0 | 56 |
| age-3.pdf | 0 | 215 |

The old `SECTION_PATTERN` (סעיף/פרק/digit-dot) found **zero** headings in the
real circular DOCX and the age-track PDF, because those documents use
unnumbered headings (כללי, הגדרות, טיפול בבקשת העברה…). The new legal parser
recovers real structure: 56 headings in the DOCX and 215 in the PDF, including
the meaningful legal sections (טיפול בבקשת העברה, מודל השקעות ברירת מחדל,
מסלולים למקבלי קצבה).

## 4. Decision

- Keep PyMuPDF + python-docx for extraction on documents that pass integrity
  checks.
- The legal hierarchy parser replaces flat regex splitting for retrieval
  structure.
- Re-extraction/OCR remains out of scope for this iteration; flagged documents
  (8, 9, 25, 27) are listed for human review before their evidence is used in
  production answers.

## Artifacts

- `spikes/legal_extraction/measure_extractors.py`
- `spikes/legal_extraction/compare_parser.py`
- `results/extractor_measurements.json`
- `results/parser_comparison.json`
- `services/legal_parser.py`
- `tests/test_legal_parser.py`, `tests/test_legal_parser_real.py`
