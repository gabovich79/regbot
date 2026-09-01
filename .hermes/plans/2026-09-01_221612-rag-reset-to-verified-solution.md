# RegBot RAG Reset Plan — From Retrieval Tuning to Verified Solution

> **For Hermes:** This is a reset plan. Do not implement or change production until Gate 0 and Gate 1 are reviewed. Use strict TDD for code changes. Do not add another global score/boost experiment before completing the evidence audit.

**Goal:** להגיע לפתרון RAG רגולטורי עובד ומאומת — לא רק לשפר Recall של document IDs — באמצעות סגירת שרשרת מלאה: מקור → metadata → document routing → section/span evidence → sufficiency gate → תשובה מנומקת ומצוטטת.

**Current reality:** ה־challenger עבר את מקרי ה־tuning אך עדיין נכשל בהכללה:

```text
Tuning:    7/7
Held-out:  3/4
Legacy:   14/17
Promotion: false
```

The current best configuration is:

```text
candidate union depth: 5
node RRF weight: 1
catalog RRF weight: 1
```

It retrieves the important repaired cases:

```text
section 25 → document 18
section 8(d) → document 36
transfer multi-source → documents 15 + 22
```

But it still fails:

```text
heldout-amendment-15
fund-mobility
identity-verification
severance-continuity
```

Production remains untouched and must remain untouched.

---

## למה נכנסנו לסבבים בלי פתרון

### 1. מדדנו document recall לפני שסגרנו את gold וה־provenance

חלק מה־legacy mapping נבנה ידנית מתוך titles ולא מתוך evidence מאומת. לכן failure יכול להיות:

- retriever failure;
- profile/title failure;
- wrong expected document;
- duplicate source identity;
- extraction failure.

בלי להפריד ביניהם, כל שינוי ranking הוא ירייה בחשיכה.

### 2. ה־prototype ערבב ארבעה תפקידים

אותו score ניסה לפתור יחד:

```text
entity identification
semantic retrieval
legal section retrieval
multi-source coverage
```

אלה שלבים שונים. RRF לא אמור להחליט לבדו שחוק, חוזר ותקנה הם אותו סוג מועמד.

### 3. בדיקות ה־tuning היו חזקות יותר מבדיקות ה־generalization

ה־7 cases פתרו את מקרי 18/22/36, אבל לא הוכיחו שהמערכת יודעת לנתב כל מסמך בקורפוס. ה־held-out וה־legacy חשפו את זה.

### 4. עצרנו ב־retrieval, בעוד שהמוצר דורש answer correctness

גם `document_recall=1.0` לא מוכיח:

- שהסעיף הנכון נשלף;
- שכל fact נדרש קיבל evidence;
- שהמערכת יודעת abstain;
- שהתשובה מסנתזת כמה מקורות נכון;
- שה־citation תומך בטענה הספציפית.

### 5. לא הגדרנו נקודת עצירה מוקדמת

מכאן והלאה: אם Gate נכשל, לא מוסיפים עוד weight או boost. חוזרים לשכבת המקור שמסבירה את הכשל.

---

# Gate 0 — Freeze and diagnostic inventory

**מטרה:** לעצור את הסחרור וליצור תמונת אמת לכל failure.

## תוצרים

Create:

```text
results/rag_reset_baseline.json
results/failure_diagnostic_matrix.json
results/RAG_RESET_REVIEW.md
```

## לכל 4 מקרי הכשל יש לתעד

```text
query
expected document/section/facts
stored profile title and official number
identity evidence lines
extracted text character/token count
available nodes/headings
lexical rank
catalog rank
dense rank
node rank
candidate union
fused rank
reason classification
```

Reason classification must be exactly one of:

```text
GOLD_ERROR
IDENTITY_PROFILE_ERROR
EXTRACTION_ERROR
NODE_PARSER_ERROR
DOCUMENT_RETRIEVAL_ERROR
SECTION_RETRIEVAL_ERROR
```

## עצירה

Gate 0 אינו משנה קוד. אם expected document אינו נתמך במקור/manifest, מתקנים את gold — לא את ה־retriever.

---

# Gate 1 — Corpus and identity correctness

**מטרה:** לגרום ל־catalog לדעת מה כל מסמך הוא לפני שמפעילים dense retrieval.

## בדיקה לכל 36 המסמכים

לכל document חייבים להיות:

```text
canonical_title
official_number where present
issuer
document_type
source/provenance
checksum
identity_evidence
integrity_status
```

## כללי זהות

- שם חוק/פקודה/תקנות/חוזר חייב להילקח משורת זהות מקורית, לא מתקציר שהומצא.
- מספר רשמי חייב להילקח מהכותרת/שורת החוזר, לא מאזכור היסטורי בגוף.
- קובץ עם title/body mismatch אינו `verified`.
- מסמך 22 נשאר מסמך ניוד; אין למחוק או לארכב אותו.
- מסמך 25 נשאר `warning` עד review; אין “לתקן” metadata אוטומטית.
- duplicate legal sources (למשל חוק 18/7/16) מקבלים source identity נפרדת ו־authority/lifecycle metadata; לא בוחרים אחד לפי ID.

## תוצר

```text
results/document_identity_review.json
results/DOCUMENT_IDENTITY_REVIEW.md
```

## Acceptance

```text
36 documents have a resolvable identity
0 silent title/body mismatches
all 4 current failures have a classified root cause
```

אם Gate 1 נכשל — לא ממשיכים ל־ranking.

---

# Gate 2 — Replace global fusion with typed routing

**מטרה:** להפריד entity routing מ־semantic retrieval ומ־legal evidence retrieval.

## Query representation

Create a deterministic query intent object:

```json
{
  "raw_question": "...",
  "entities": [
    {
      "text": "חוק הפיקוח על שירותים פיננסיים (קופות גמל)",
      "entity_type": "law",
      "normalized_terms": ["..."],
      "exact_official_number": null
    }
  ],
  "section_refs": ["25"],
  "operations": ["העברה", "שעבוד"],
  "source_roles": ["law", "regulation", "circular"],
  "multi_source": true
}
```

## Routing rules — generic, not question-specific

1. Explicit official number → exact catalog candidate.
2. Explicit law/regulation/circular title → catalog/entity candidate.
3. Explicit section reference → section search only within routed candidates.
4. No explicit source entity → hybrid document retrieval.
5. Multi-source intent → preserve candidates per required source role; do not let one fused top-5 erase a complementary authority.

## Candidate policy

Use:

```text
catalog/entity candidates
∪ profile lexical candidates
∪ profile dense candidates
∪ node lexical candidates
∪ node dense candidates when available
→ typed candidate set
→ bounded rerank within type/authority constraints
```

The candidate set and final context set must be separate. Recall candidates can be larger than the final evidence packet.

## Required tests

Create/modify:

```text
tests/test_query_routing.py
tests/test_catalog_entity_matching.py
tests/test_multisource_candidate_coverage.py
```

Tests must cover:

- named law with nonexistent section → correct law candidate, later abstention;
- named circular → correct circular candidate;
- two required source roles → both survive candidate formation;
- generic semantic question → no catalog false positive;
- duplicate legal source → authority/lifecycle policy is deterministic.

## Acceptance

The four previously failing cases must be corrected by their classified root causes, not by document-ID exceptions.

---

# Gate 3 — Exact section/span retrieval and evidence gate

**מטרה:** לעבור מ־“מצאנו מסמך” ל־“יש לנו ראיה שמותר לנסח ממנה תשובה”.

## Evidence packet

Every evidence block must include:

```text
stable evidence ID
document ID
canonical title
official number
source URL/checksum
section/node path
page range when source provenance supports it
raw source text
```

Profile summaries and generated descriptions may assist routing but cannot be cited as legal evidence.

## Required-fact coverage

For every case, calculate:

```text
required_facts
covered_facts
unsupported_facts
required_sections
found_sections
missing_sections
source_role_coverage
```

## Abstention contract

For `section 999`:

```text
route to the correct law document
find no section 999
abstain clearly
never use section 998/nearby text as a substitute
```

## Multi-source contract

For transfer:

```text
source 15 → regulatory framework/conditions
source 22 → implementation procedure/timelines
```

A response is not sufficient if it cites only one of the two when both roles are required.

## Tests

Create:

```text
tests/test_evidence_sufficiency_gate.py
tests/test_claim_bound_citations.py
tests/test_multisource_synthesis_contract.py
```

---

# Gate 4 — Evaluation and shadow decision

**מטרה:** להחליט אם יש מוצר עובד, לא אם יש prototype מעניין.

## Evaluation partitions

```text
Tuning:    7 hierarchical cases
Held-out: 4+ independent cases
Legacy:   17 existing cases
```

Add at least 3 more held-out cases after identity review, including:

- exact official circular number;
- duplicate-law source selection;
- multi-source question with complementary authority;
- missing-section abstention.

## Promotion criteria

All must pass:

```text
Tuning all-required-document@5 = 1.0
Tuning all-required-document@3 ≥ 0.857
Held-out all-required-document@5 = 1.0
Legacy: no unexplained failures and no material regression
Section/span recall meets case expectations
Required-fact coverage = 1.0 for non-abstention cases
Abstention correctness = 1.0 for missing-evidence cases
Citation entailment/completeness passes human review
```

Until then:

```text
production answer path unchanged
no feature flag
no production reindex
no shadow answer generation
```

## Shadow phase — only after all gates

Shadow must compare, per query:

```text
flat selected evidence
challenger selected evidence
required-fact coverage
citation coverage
abstention decision
latency
token/context size
```

Shadow is read-only and produces a review sheet. It must not silently alter user answers.

---

# Implementation order

```text
1. Gate 0 failure matrix
2. Gate 1 identity/provenance audit
3. Fix gold/profile/extraction findings
4. Gate 2 typed query routing
5. Re-run 7 + held-out + legacy retrieval
6. Gate 3 section/span + evidence gate
7. End-to-end answer evaluation
8. Only then read-only shadow
```

## Explicit stop rule

If the next implementation pass fails any Gate 0–3 criterion:

```text
stop coding
record evidence
classify root cause
review architecture
```

No fourth round of blind ranking tweaks.

## Files likely to change

```text
services/document_profile_service.py
services/document_retriever.py
services/section_retriever.py
services/legal_parser.py
scripts/measure_challenger_hybrid.py
scripts/measure_challenger_gate.py
scripts/run_challenger_ablations.py
scripts/run_local_challenger.sh

eval/hierarchical_cases.jsonl
eval/hierarchical_cases_heldout.jsonl
eval/retrieval_cases.jsonl

tests/test_document_profiles_hierarchical.py
tests/test_document_retriever.py
tests/test_section_retriever.py
tests/test_query_routing.py
tests/test_evidence_sufficiency_gate.py
```

## What is deliberately not planned

- no production change;
- no new vector database;
- no more per-query boosts;
- no hand-coded document IDs;
- no prompt rewrite before evidence packets are correct;
- no claim that a passing document metric means the answer is legally correct.
