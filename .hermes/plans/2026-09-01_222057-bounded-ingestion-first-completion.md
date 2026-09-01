# RegBot Bounded Ingestion-First Completion Plan

> **Mode:** plan only. No production change, corpus mutation, upload, reindex, or code implementation is performed by this document.

**Goal:** לסיים את פיתוח ה־RAG דרך חוזה ingestion מחייב: כל מסמך נכנס פעם אחת ויוצא רק כאשר הוא מקור מאומת, מפורק, מאונדקס ומוכן לראיה. אין עוד ניסיונות להציל retrieval על corpus שחלק ממנו אינו מתועד או אינו מובנה.

**Definition of Success:**

```text
Every active document has a verified ingestion receipt.
Every retrieval candidate resolves to a verified document + section/node.
Every answer-bearing fact resolves to raw source evidence.
A finite, measured evaluation gate passes before production activation.
```

**Bounded execution:** exactly **three implementation rounds**. A round cannot spawn an unplanned fourth tuning round. If a source fails, it becomes a finite re-upload item for Guy—not an algorithmic guessing game.

---

# The missing contract: ingestion must finish the job

At upload/re-ingestion, a document must complete this state machine:

```text
uploaded
→ original_preserved
→ extracted
→ identity_verified
→ structured
→ indexed
→ validated
→ active
```

It may instead end only in:

```text
needs_reupload
needs_human_review
failed_with_reason
```

**It may never be `active` while any required stage is absent.**

## Per-document ingestion receipt

Every active document receives a durable receipt containing:

```text
source artifact path + SHA-256
source URL / uploader provenance
file type + page count where applicable
extracted character count + token count
extraction health (RTL, binary, empty, encoding)
canonical title
official number
issuer
legal document type
lifecycle / effective date
identity evidence lines from raw source
heading outline
node count + section paths
keywords (deterministic from title / number / headings)
profile text hash
FTS index record count
embedding record count + model/dimension/index version
integrity status
validation timestamp
```

No LLM-generated summary may serve as identity or legal evidence. It can be retrieval metadata only.

---

# Round 1 — Build the ingestion contract and prove it on one document

**Scope:** code and local challenger only. No production route changes.

## 1.1 Implement one atomic ingestion pipeline

Create or extend one service that accepts an original document and produces all artifacts in one transaction/workflow:

```text
original source
→ extraction
→ identity/profile
→ headings + legal nodes
→ deterministic keywords
→ FTS document index + FTS node index
→ embeddings
→ integrity validation
→ ingestion receipt
```

If any stage fails:

```text
rollback / do not activate document
persist failure reason
mark document needs_reupload or needs_human_review
```

## 1.2 Required validations

A document passes only if all are true:

```text
original exists and checksum recorded
extracted text is non-empty and non-binary
Hebrew RTL extraction is not reversed/garbled
canonical title has source evidence
stored title/body overlap is sufficient or human-approved
at least one meaningful legal node exists for structured sources
all generated nodes have token-safe retrieval text
all FTS/embedding counts equal expected counts
```

## 1.3 Pilot source

Use one known-good source with clear structure, preferably:

```text
Document 22 — העברת כספים בין קופות גמל
```

Why: it has a known official number, clear headings, and existing transfer evidence.

## 1.4 Round-1 exit gate

```text
1 document has a complete receipt
all artifact counts reconcile
node tree can retrieve the known transfer procedure section
receipt is human-readable in the admin/report output
all new tests pass
```

If this fails, fix ingestion only. Do not touch ranking.

---

# Round 2 — Normalize the corpus, document by document where necessary

**Scope:** rebuild the challenger corpus from the ingestion contract, then produce a finite review queue.

## 2.1 Re-ingest all 36 sources locally

Run the same pipeline over every source. There are only three outcomes per document:

| Status | Meaning | Action |
|---|---|---|
| `active` | complete receipt and validation pass | include in challenger corpus |
| `needs_human_review` | source valid, identity/extraction ambiguous | review one short evidence card |
| `needs_reupload` | original/extraction unusable or source missing | Guy uploads that exact document once |

## 2.2 No silent repair

Examples:

- Document 25 title/body mismatch → `needs_human_review` or `needs_reupload`, never auto-renamed.
- Binary/RTL failures → `needs_reupload` with the exact document ID and reason.
- Duplicate laws → retain source identities and lifecycle metadata; do not choose by incidental DB ID.

## 2.3 User interaction is bounded

Guy sees one finite review sheet with only exceptions:

```text
Document ID
current title
identity evidence from source
problem class
required action: approve / replace / re-upload
```

For a failed source, one re-upload is attempted. If it still fails, it remains excluded with an explicit reason. It cannot silently contaminate active retrieval.

## 2.4 Round-2 exit gate

```text
Every active document has a complete ingestion receipt
No active document has empty/binary/reversed extraction
No active document has unresolved title/body mismatch
Every active document has nodes + keywords + FTS + embeddings
The exception queue is empty or explicitly excluded from active corpus
```

The result is a stable corpus manifest generated from receipts—not a manually guessed mapping.

---

# Round 3 — Build the actual answer-capable retrieval path and finish

**Scope:** one final challenger implementation + one correction pass maximum.

## 3.1 Retrieval architecture (fixed)

```text
A. Query parsing
   - exact law/circular/number/section entities
   - operation/topic terms
   - explicit multi-source intent

B. Document routing
   - exact catalog identity match
   - lexical profile retrieval
   - dense profile retrieval
   - candidate union, typed by source role

C. Section retrieval
   - only inside routed documents
   - exact section / heading match first
   - lexical + dense node retrieval
   - parent expansion to minimum legal context

D. Evidence sufficiency gate
   - every required fact maps to raw node/source span
   - missing section/fact → abstain
   - multi-source question requires coverage of each source role

E. Generation
   - reasoned Hebrew answer
   - claim-bound citations to raw evidence only
```

## 3.2 Fixed evaluation set

Use only a verified corpus-derived set:

```text
7 tuning cases
7 held-out cases
17 legacy cases, regenerated from receipt-backed identities
```

Each case carries:

```text
required document(s)
required section(s)
required fact(s)
required source role(s)
expected abstention where relevant
```

## 3.3 One correction pass only

1. Run the full evaluation.
2. Classify every failure as one of:

```text
SOURCE / EXTRACTION / IDENTITY / ROUTING / SECTION / EVIDENCE / GOLD
```

3. Apply one correction pass at the classified layer.
4. Re-run once.

If the final gate still fails, development is not extended by another random tuning round. The output is a concrete finite blocker list:

```text
exact source documents to re-upload
exact gold cases to correct
exact architectural layer that remains inadequate
```

## 3.4 Final promotion gate

All conditions must pass:

```text
Active corpus: 100% receipt-backed
Document recall: 100% required documents @5
Section recall: 100% on exact-section cases
Required-fact coverage: 100% on answerable cases
Abstention correctness: 100% on missing-evidence cases
Legacy set: no unexplained regression
Citations: source/section-resolvable
```

Only then:

```text
read-only shadow against production
→ human review sheet
→ explicit approval
→ production activation
```

---

# Non-negotiable rules

```text
No production upload/reindex during rounds 1–3
No global “try another RRF weight” cycles
No document-ID boosts
No question-rule table
No auto-metadata repair
No answer generation before evidence gate
No active document without a receipt
```

# Files likely to change

```text
services/document_ingestion_service.py          (create)
services/document_profile_service.py
services/document_integrity_service.py
services/legal_parser.py
services/document_retriever.py
services/section_retriever.py
services/evidence_gate_service.py               (create)
models/database.py
main.py                                          (only after challenger passes)

scripts/reingest_challenger_corpus.py           (create)
scripts/build_ingestion_review_sheet.py         (create)
scripts/run_final_challenger_gate.py            (create)

tests/test_document_ingestion_contract.py       (create)
tests/test_ingestion_receipt.py                 (create)
tests/test_evidence_gate_service.py             (create)
tests/test_final_challenger_gate.py             (create)
```

# Completion statement

This plan ends in one of two honest, finite outcomes:

1. **Success:** a verified corpus and answer-capable challenger pass the final gate and are ready for shadow review.
2. **Explicit source blocker:** a finite list of exact documents that must be uploaded/replaced once before final completion.

It does not permit a third outcome called “let’s try one more ranking tweak.”
