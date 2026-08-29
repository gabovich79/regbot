# Hierarchical Legal RAG Migration Implementation Plan

> **For Hermes:** Implement this plan task-by-task with strict TDD. Do not alter the production answer path until the challenger passes the decision gates in Tasks 8–9.

> **Status: production plan — not yet implemented.**

**Goal:** להחליף את מנגנון ה־flat chunk retrieval של RegBot במנגנון היררכי: זיהוי מסמך → חיפוש סעיף בתוך המסמך → הרחבה ל־parent section → evidence gate → תשובה מצוטטת.

**Architecture:** נשמור את המקורות, עמודי ה־PDF, תוקף המסמכים וחוזה הציטוטים שכבר נבנו. נוסיף קטלוג מסמכים קנוני ועץ nodes משפטי. החיפוש יהיה דו־שלבי עם dense + lexical + RRF בכל שלב. summaries/contextual headers ישמשו retrieval בלבד; התשובה תראה רק raw source evidence.

**Tech Stack:** Python 3.11, FastAPI, SQLite + FTS5, NumPy exact vector scoring, OpenAI `text-embedding-3-large` כ־baseline, Gemini generation, optional Haystack POC, optional BGE-M3 ablation. אין Qdrant/Postgres/RAGFlow בגרסה הראשונה.

## חוזה התוצר הסופי

RegBot אינו מנוע חיפוש או מנוע ציטוטים. התוצר הסופי הוא תשובה מקצועית, מנומקת ומסונתזת, כאשר citations הם שכבת ההוכחה והביקורת.

לכל שאלה המערכת צריכה:

1. לפרק את השאלה לסוגיות משנה כאשר נדרש;
2. לאתר לכל סוגיה את המסמך והסעיף בעלי הסמכות המתאימה;
3. להרכיב evidence coverage map שמראה אילו מקורות תומכים בכל חלק בתשובה;
4. לסנתז כלל, תנאים, חריגים, תחולה, תוקף והשלכה מעשית;
5. להבחין בתפקידי המקורות: חוק/פקודה כבסיס, תקנות כפירוט, חוזר כיישום, ופרמטר שנתי כערך עדכני;
6. ליישב מקורות משלימים, ולהציג סתירה או אי־ודאות במקום להחליק אותן;
7. לענות בשפה מקצועית וברורה, לא באמצעות הדבקת ציטוטים;
8. לצרף citation לכל טענה מהותית כך שהקורא יכול לבדוק את הסינתזה;
9. להימנע ממסקנה כאשר evidence coverage אינו מספיק.

דוגמת flow לשאלה רב־מקורית:

```text
פקודה/חוק — המסגרת המשפטית
+ תקנות — תנאים וחריגים
+ חוזר — אופן היישום
+ פרמטר שנתי — ערך מעודכן
→ synthesis מקצועי אחד
→ claim-level citations
```

קריטריון הצלחה אינו רק שהמקורות הנכונים נשלפו. נדרש שגם generator:

- ישתמש בכל העובדות ההכרחיות;
- לא ישמיט תנאי load-bearing;
- לא יציג תקרת רגולציה כאישור אישי;
- לא יערבב תקופות תוקף;
- לא יסתור מקור ללא הצגת הסתירה;
- יפיק תשובה עצמאית, מנומקת וקוהרנטית.

---

## 1. מסקנות המחקר

### 1.1 Retrieval קצר ושטוח מאבד הקשר

RAPTOR ו־LongRAG מזהים את אותה בעיה שרואים ב־RegBot: יחידות קצרות מאבדות הקשר ויוצרות hard negatives. RAPTOR בונה עץ רב־רמות; LongRAG מגדיל את יחידת ה־retrieval ומעביר יותר עבודה ל־reader.[1][2]

**מסקנה ל־RegBot:** לא לשלוח מסמך שלם ולא להישאר עם leaf chunks מבודדים. לבצע:

```text
Document profile → leaf retrieval → parent section expansion
```

### 1.2 Contextual headers + hybrid retrieval

Contextual Retrieval מציע לצרף לכל chunk הקשר מסמכי לפני embedding ו־BM25.[3] Qdrant מתעד את התועלת בשילוב dense ו־sparse, ואת RRF למיזוג התוצאות.[10]

**מסקנה:** embedding text יכלול header דטרמיניסטי:

```text
מסמך | מנפיק | סוג | מספר רשמי | פרק | סעיף | תוקף | עמוד | נושא
```

אך `raw_text` נשאר ללא שינוי והוא היחיד שמותר לצטט.

### 1.3 Hierarchical parent/child retrieval הוא דפוס מוכן

Haystack מספק `HierarchicalDocumentSplitter` ו־`AutoMergingRetriever`; LlamaIndex מספק Auto Merging שמאחד leaves חזרה להורה.[12][13][14]

**מסקנה:** אין צורך להמציא את הדפוס. ב־POC נשווה implementation קטן ומפורש מול Haystack. production יאמץ את המימוש שעובר את המדדים, לא את ה־framework עם הלוגו היפה ביותר.

### 1.4 משפטי דורש precision ברמת span/section

LegalBench-RAG מדגיש שליפה של מקטעים מינימליים ורלוונטיים, ולא רק document IDs או רצפים גדולים ולא מדויקים.[4]

**מסקנה:** document-first אינו document-only. המסמך מצמצם את מרחב החיפוש; הראיה הסופית היא סעיף/תת־סעיף מדויק. Parent expansion מוגבל לסעיף המשפטי, לא למסמך כולו.

### 1.5 Summaries אינם ראיה

RAPTOR משתמש בסיכומים רקורסיביים לשיפור retrieval.[1] ב־RegBot, summary יכול לעזור לבחור מסמך או לענות על שאלות הוליסטיות, אך הוא חומר סינתטי.

**כלל:** `profile_summary`, `scope_in`, `scope_out` ו־topic summaries אינם נכנסים ל־evidence packet ואינם מקבלים citation IDs.

### 1.6 Reranker ו־embedding חדש הם ablation, לא אמונה

BGE-M3 תומך ביותר מ־100 שפות וב־dense, sparse ו־multi-vector retrieval; ColBERTv2 מיישם late interaction ברמת token.[6][7]

**מסקנה:** לבדוק:

```text
A: existing OpenAI dense + FTS5 BM25 + RRF
B: BGE-M3 dense + sparse
C: B + BGE reranker
D: ColBERT late interaction (רק אם A–C אינם מספיקים)
```

אין הנחה שאיכות עברית משפטית טובה רק כי המודל multilingual.

### 1.7 Evaluation חייב להפריד retriever מ־generator

RAGAS ו־RAGChecker מפרידים איכות context, faithfulness ואיכות generation; ALCE מודד גם citation quality.[5][8][9] LegalBench-RAG מספק השראה למדידה ברמת span.[4]

**מסקנה:** לא למדוד “התשובה נשמעת טוב”. למדוד בנפרד:

```text
document recall
section/span recall
context precision
required-fact coverage
citation entailment/completeness
abstention
freshness/validity
```

---

## 2. החלטת ארכיטקטורה

### 2.1 הבחירה המומלצת

```text
SQLite documents + source artifacts
→ document_profiles
→ hierarchical document_nodes
→ SQLite FTS5 + exact dense scoring
→ RRF document retrieval
→ scoped RRF section retrieval
→ parent section expansion
→ evidence sufficiency gate
→ Gemini answer + citations
```

SQLite FTS5 מספק full-text search בתוך המסד הקיים.[16] עם 36 מסמכים ו־819 chunks כיום, exact vector scan עדיין זול ופשוט. אין הצדקה להוסיף שירות vector DB לפני שמדד או עומס מוכיחים צורך.

### 2.2 מה לא נבחר כרגע

| חלופה | החלטה | סיבה |
|---|---|---|
| כל הקורפוס בפרומפט | לא | הקורפוס ~2.3M tokens; גישת full-context מתאימה ל־KB קטן בהרבה.[3] |
| flat global chunk search | להחליף | הוכח כרגיש לכותרות שגויות ול־context loss |
| RAPTOR מלא | מאוחר יותר | summaries טובים לניווט, אך מסוכנים כראיה משפטית |
| GraphRAG | לא כעת | הבעיה העיקרית היא מקור/סעיף, לא גרף ישויות |
| Qdrant | defer | storage/search engine; אינו מתקן ingestion או hierarchy בעצמו |
| PostgreSQL + pgvector | defer | יעד migration אפשרי כשהיקף/תחרותיות יצדיקו; pgvector ניתן לשילוב עם PostgreSQL FTS.[11] |
| RAGFlow | לא production | operationally כבד יחסית; דרישותיו המוצהרות מתחילות ב־4 cores, 16GB RAM ו־50GB disk.[17] |
| Haystack | challenger POC | hierarchy/auto merge מוכנים; ייבחן בלי לחייב framework migration |
| LlamaIndex | reference/POC | AutoMerging טוב, אך אין צורך להכניס שני frameworks במקביל |

---

## 3. מודל הנתונים החדש

### 3.1 `document_profiles`

```sql
document_id PK
canonical_title
official_number
issuer
document_type
publication_date
effective_date
valid_until
lifecycle_status
supersedes_document_id
profile_summary
scope_in_json
scope_out_json
topics_json
keywords_json
heading_outline_json
identity_evidence_json
profile_embedding
integrity_status       -- verified | warning | failed | pending
integrity_reasons_json
review_status          -- machine | human_verified
profile_version
created_at
updated_at
```

### 3.2 `document_nodes`

```sql
id PK
document_id
parent_id
node_type              -- document | chapter | section | subsection | paragraph | table
node_path
section_label
heading
raw_text
retrieval_text         -- contextual header + raw text
page_start
page_end
ordinal
text_hash
embedding
is_evidence
index_version
```

### 3.3 FTS5

```sql
document_profiles_fts(canonical_title, official_number, issuer, topics, keywords, profile_summary)
document_nodes_fts(heading, section_label, retrieval_text)
```

Profile summary ו־retrieval text הם index material. `raw_text` + source metadata הם evidence material.

---

## 4. Query flow החדש

```text
1. Parse query
   - exact circular/law/section/year
   - product/entity
   - operation/issue
   - historical/current intent

2. Retrieve documents
   - exact metadata match
   - FTS5/BM25 profile search
   - profile embedding search
   - authority + validity filtering
   - RRF → top 3–5 documents

3. Retrieve sections inside selected documents only
   - exact section match
   - FTS5/BM25 node search
   - dense node search
   - RRF → top leaves

4. Parent expansion
   - expand matched leaf to complete legal section
   - merge siblings only above threshold
   - preserve minimal relevant parent section

5. Evidence sufficiency gate
   - source/section/page resolvable
   - exact section queries require exact section evidence
   - numeric claims require matching raw evidence
   - unresolved source conflict → abstain/clarify

6. Generation
   - raw evidence blocks only
   - every material claim cites evidence ID
   - profile summaries never cited

7. Post-generation validation
   - citation IDs exist in context
   - citations entail claims
   - no unsupported number/date
   - no `THOUGHT`/tool trace
```

---

## 5. תכנית יישום

### Task 0: Freeze baseline and create branch

**Objective:** לקבע את production הנוכחי ולהכין challenger מבודד.

**Files:**
- Create: `eval/hierarchical_cases.jsonl`
- Create: `results/hierarchical_baseline.json`
- Create: `docs/hierarchical-rag-architecture.md`

**Steps:**
1. Create branch `feat/hierarchical-legal-retrieval`.
2. Record current commit, models, 36-document manifest, 819 chunks and current metrics.
3. Export read-only retrieval results for all existing eval cases.
4. Record the two corrected production cases: age tracks and fund transfer.
5. Commit baseline only.

**Verification:** existing `pytest`, `compileall`, `git diff --check`, `hermes verify` pass.

---

### Task 1: Add profile/node schema atomically

**Objective:** להוסיף data model בלי לשנות retrieval קיים.

**Files:**
- Modify: `models/database.py`
- Create: `tests/test_hierarchical_migrations.py`

**TDD:**
1. RED: migrations create `document_profiles`, `document_nodes`, both FTS5 tables and indexes.
2. GREEN: additive/idempotent migrations only.
3. Verify existing DB still boots and existing chunks remain unchanged.

**Commit:** `feat: add hierarchical retrieval schema`

---

### Task 2: Build document profiles and corpus-integrity audit

**Objective:** להבין ולוודא כל מסמך פעם אחת לפני retrieval.

**Files:**
- Create: `services/document_profile_service.py`
- Create: `services/document_integrity_service.py`
- Create: `scripts/build_document_profiles.py`
- Create: `tests/test_document_profiles.py`
- Create: `tests/test_document_integrity.py`
- Modify: admin UI to display integrity status and reasons

**Behavior:**
- Deterministic extraction of title/number/date/issuer from source text.
- Optional LLM structured summary for `profile_summary`, `scope_in`, `scope_out`; output schema validated.
- Compare stored title/topic with title page, headings and body.
- Flag mismatch; never silently rename/archive/delete.
- Profile fields retain `identity_evidence` with source page/quote.

**Required regression:** the former document-22 title/content mismatch must produce `warning` or `failed`, not `verified`.

**Run:**
```bash
python scripts/build_document_profiles.py --dry-run
python scripts/build_document_profiles.py --document-id 22
```

First command is read-only and prints proposed profiles/flags. Second writes one representative profile only after review.

**Commit:** `feat: profile and validate regulatory documents`

---

### Task 3: Compare extraction/parsing on representative documents

**Objective:** לבחור parser לפי תוצאות עבריות אמיתיות.

**Files:**
- Create: `spikes/legal_extraction/compare_extractors.py`
- Create: `spikes/legal_extraction/results.json`
- Create: fixtures/tests for five source types

**Corpus:**
1. Hebrew law PDF with numbered sections.
2. Circular DOCX.
3. Scanned/OCR-poor PDF.
4. Table/appendix-heavy PDF.
5. Current age-track PDF.

**Variants:**
- Existing PyMuPDF/python-docx.
- Docling challenger, which preserves hierarchy, layout and provenance.[15]

**Metrics:** heading recall, section-label recall, page provenance, table retention, RTL reading order, processing time.

**Decision:** use Docling only for classes where it measurably wins; do not replace working extraction globally.

---

### Task 4: Implement legal hierarchy parser

**Objective:** ליצור עץ דטרמיניסטי של מסמך/פרק/סעיף/תת־סעיף.

**Files:**
- Create: `services/legal_parser.py`
- Create: `tests/test_legal_parser.py`
- Modify: `services/rag_service.py` only to call the new indexer, not retrieval yet

**TDD cases:**
- `8(ד)` retains 50%, 80%, 7 years in the same parent section.
- section 25 retains transfer, pledge and garnishment.
- 9(16א)/(16ב) remain separate nodes.
- page boundaries survive.
- tables/appendices become typed nodes.
- unstructured document gets safe paragraph fallback.

**Commit:** `feat: parse Hebrew legal documents into hierarchy`

---

### Task 5: Index profile and nodes

**Objective:** לייצר שני indexes ולשמור swap אטומי.

**Files:**
- Create: `services/hierarchical_index_service.py`
- Create: `tests/test_hierarchical_indexing.py`
- Modify: `models/database.py`
- Add CLI: `scripts/reindex_hierarchical.py --dry-run|--document-id|--all`

**Rules:**
- `retrieval_text = deterministic contextual header + raw_text`.
- Embed profile and nodes separately.
- Populate FTS5 transactionally.
- Build complete next `index_version` before activating it.
- Failure/cancellation leaves current flat index and prior hierarchical version intact.

**Commit:** `feat: index document profiles and legal nodes`

---

### Task 6: Document-level retriever

**Objective:** לבחור מסמכים לפני סעיפים.

**Files:**
- Create: `services/document_retriever.py`
- Create: `tests/test_document_retriever.py`

**Pipeline:** exact identifiers + FTS5 BM25 + dense profile score + generic authority/validity filters → RRF.

**No hardcoded topic rules.** Query decomposition is generic, logged and schema-bound.

**Critical tests:**
- age-track question → new 2024 document in top 3.
- transfer question → renamed transfer circular in top 3.
- section 25 → provident-fund law, not regulations with an unrelated section 25.
- historical question may retrieve superseded source; current question prefers current source.

**Commit:** `feat: retrieve regulatory documents before sections`

---

### Task 7: Scoped section retriever and parent expansion

**Objective:** לחפש raw legal evidence רק בתוך selected documents.

**Files:**
- Create: `services/section_retriever.py`
- Create: `tests/test_section_retriever.py`
- Create: `tests/test_parent_expansion.py`
- Optional challenger: `spikes/haystack_hierarchy/`

**Pipeline:** exact section + FTS5 + dense → RRF → optional rerank → parent section expansion.

**Rules:**
- top document selection does not count as evidence.
- leaf matches expand only to the legal parent needed for complete meaning.
- prevent unrelated neighboring sections from entering context.

**Commit:** `feat: retrieve and expand legal sections within selected documents`

---

### Task 8: Evidence sufficiency and citation validation

**Objective:** למנוע answer generation כשאין בסיס מספק.

**Files:**
- Create: `services/evidence_gate.py`
- Create: `services/citation_validator.py`
- Create: `tests/test_evidence_gate.py`
- Create: `tests/test_citation_validator.py`
- Modify: `services/claude_service.py`

**Tests:**
- explicit section missing → abstain, do not answer from nearby section.
- title matches but body does not → reject document.
- unsupported numeric claim → validation failure/LOW confidence.
- cited ID absent from context → validation failure.
- `THOUGHT:` or tool trace → stripped/rejected before delivery.

**Commit:** `feat: gate regulated answers on sufficient evidence`

---

### Task 9: Evaluation and ablation

**Objective:** להחליט לפי evidence retrieval, לא לפי תחושת שיחה.

**Files:**
- Extend: `eval/hierarchical_cases.jsonl`
- Create: `scripts/run_hierarchical_evaluation.py`
- Create: `scripts/compare_retrievers.py`
- Create: `results/retrieval_ablation.json`

**Gold case fields:**
```json
{
  "question": "...",
  "required_documents": [38],
  "required_sections": ["מודל השקעות תלוי גיל"],
  "required_facts": ["50 ומטה", "50 עד 60", "60 ומעלה"],
  "forbidden_sources": [22],
  "forbidden_claims": [],
  "expected_abstention": false,
  "human_verified": true
}
```

**Metrics:**
- Document Recall@3/5 and MRR.
- Section/span Recall@5/10 and precision.
- Context minimality/noise.
- Required-fact coverage.
- Multi-source coverage: כל מקור load-bearing הנדרש לשאלה אכן שימש בתשובה.
- Synthesis completeness: כלל, תנאים, חריגים, תחולה והשלכה מעשית מופיעים כאשר נדרשים.
- Source-role correctness: חוק/תקנות/חוזר/פרמטר שנתי משמשים בתפקיד הנכון.
- Cross-source consistency: אין ערבוב תוקף או סתירה שלא הוצגה.
- Citation correctness and completeness.
- Unsupported-number rate.
- Abstention precision/recall.
- Freshness/supersession correctness.
- p50/p95 latency and cost.

**Ablations:** existing OpenAI vs BGE-M3; with/without contextual header; BM25/dense/RRF; with/without parent expansion; reranker variants.

**Decision gate:**
- 100% document+section success on critical high-risk cases.
- ≥95% document Recall@5 overall.
- ≥90% section Recall@10 and ≥80% evidence Precision@5.
- zero unsupported numerical claims in high-risk set.
- ≥95% citation correctness.
- no regression on accepted current cases.

Thresholds are provisional until the baseline is recorded; critical-case 100% is not negotiable.

---

### Task 10: Shadow mode

**Objective:** להשוות production ללא שינוי תשובות למשתמש.

**Files:**
- Modify: `services/rag_service.py` via retriever interface
- Create: `services/retriever_interface.py`
- Add setting: `RAG_RETRIEVER=flat|hierarchical|shadow`
- Create: `retrieval_shadow_runs` table

**Behavior:**
- flat path still answers.
- hierarchical path runs read-only in parallel.
- log selected document/node IDs, scores, latency and sufficiency result; no raw user PII beyond current logs.

**Duration:** minimum 50 real queries or one week, whichever produces enough high-risk cases.

**Commit:** `feat: compare hierarchical retrieval in shadow mode`

---

### Task 11: Controlled migration and rollback

**Objective:** להפעיל רק אחרי מעבר שער המדידה ואישור גיא.

**Steps:**
1. Enable hierarchical retrieval for admin/test traffic.
2. Re-run complete answer benchmark.
3. Enable feature flag for production.
4. Keep flat retriever rollback for one release.
5. Remove question-specific boosts and `legal_rules.py` only after their cases pass through generic retrieval.
6. Do not change generation model in the same release.

**Rollback:** one setting returns to `flat`; old index remains untouched.

---

## 6. Verification commands

```bash
pytest -q
python -m compileall -q main.py models services scripts tests
node --check frontend/static/app.js
git diff --check
hermes verify --json
```

Hierarchical checks:

```bash
python scripts/build_document_profiles.py --dry-run
python scripts/reindex_hierarchical.py --dry-run
python scripts/run_hierarchical_evaluation.py
python scripts/compare_retrievers.py --baseline flat --candidate hierarchical
```

Every command must exit non-zero on failed documents/cases; no “95% succeeded” message with a hidden zero-chunk document.

---

## 7. Review gates requiring Guy approval

1. **After Task 2:** review only flagged/mismatched document profiles.
2. **After Task 3:** approve extractor choice from measured results.
3. **After Task 9:** approve candidate architecture and embedding/reranker.
4. **After Task 10:** approve production feature flag.
5. **Before Task 11 cleanup:** approve removal of legacy special rules.

## Sources

[1] https://arxiv.org/abs/2401.18059 — RAPTOR
    > "most existing methods retrieve only short contiguous chunks from a retrieval corpus, limiting holistic understanding of the overall document context"
[2] https://arxiv.org/abs/2406.15319 — LongRAG
    > "The loss of contextual information in the short, chunked units may increase the likelihood of introducing hard negatives during the retrieval stage."
[3] https://www.anthropic.com/engineering/contextual-retrieval — Contextual Retrieval
    > "The method is called “Contextual Retrieval” and uses two sub-techniques: Contextual Embeddings and Contextual BM25."
    > "If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt that you give the model, with no need for RAG or similar methods."
[4] https://arxiv.org/abs/2408.10343 — LegalBench-RAG
    > "LegalBench-RAG emphasizes precise retrieval by focusing on extracting minimal, highly relevant text segments from legal documents."
[5] https://arxiv.org/abs/2309.15217 — RAGAS
    > "there are several dimensions to consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithful way, or the quality of the generation itself."
[6] https://arxiv.org/abs/2402.03216 — BGE-M3
    > "It can simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval."
[7] https://arxiv.org/abs/2112.01488 — ColBERTv2
    > "late interaction models produce multi-vector representations at the granularity of each token"
[8] https://arxiv.org/abs/2408.08067 — RAGChecker
    > "a fine-grained evaluation framework, RAGChecker, that incorporates a suite of diagnostic metrics for both the retrieval and generation modules."
[9] https://arxiv.org/abs/2305.14627 — ALCE
    > "We develop automatic metrics along three dimensions -- fluency, correctness, and citation quality"
[10] https://qdrant.tech/documentation/search/hybrid-queries — Qdrant Hybrid Queries
    > "it is often useful to combine dense and sparse vectors to get the best of both worlds: semantic understanding from dense vectors and precise word matching from sparse vectors."
[11] https://github.com/pgvector/pgvector/blob/master/README.md — pgvector Hybrid Search
    > "Use together with Postgres full-text search for hybrid search."
[12] https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever — LlamaIndex Auto Merging Retriever
    > "looks at a set of leaf nodes and recursively “merges” subsets of leaf nodes that reference a parent node beyond a given threshold."
[13] https://docs.haystack.deepset.ai/docs/automergingretriever — Haystack AutoMergingRetriever
    > "Use AutoMergingRetriever to improve search results by returning complete parent documents instead of fragmented chunks when multiple related pieces match a query."
[14] https://docs.haystack.deepset.ai/docs/hierarchicaldocumentsplitter — Haystack HierarchicalDocumentSplitter
    > "Use this component to create a multi-level document structure based on parent-children relationships between text segments."
[15] https://docling-project.github.io/docling/concepts/docling_document — DoclingDocument
    > "Document hierarchy with sections and groups"
[16] https://www.sqlite.org/fts5.html — SQLite FTS5
    > "FTS5 is an SQLite virtual table module that provides full-text search functionality to database applications."
[17] https://github.com/infiniflow/ragflow — RAGFlow
    > "CPU >= 4 cores"
