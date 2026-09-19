# D37 Vertical Slice — מסמך handoff מלא

> נכתב 2026-09-15. מטרה: לאפשר המשך במערכת/סוכן אחר בלי לשחזר מחדש שום החלטה, ממצא או פקודה.
> פרויקט: RegBot — RAG רגולטורי היררכי בעברית. ריפו: `/Users/guygabovich/Projects/regbot-hierarchical`, branch `feat/hierarchical-legal-retrieval`.

---

## 1. תקציר

נבנה ונבדק **vertical slice שלם על מסמך אחד (D37 = פקודת מס הכנסה [נוסח חדש])**, עבור שאלת מיסוי משיכת קרן השתלמות. המסלול עובד end-to-end, **חינמי ומקומי** (אין API בתשלום, אין ענן):

```
מקור PDF (עם checksum)
→ שליפה (local embeddings, nomic-embed via Ollama)
→ Evidence units (סעיף + עמוד + span hash)
→ תשובה (LLM מקומי מצטט [[ID]] בלבד, לא מתמלל)
→ resolver דטרמיניסטי (מצמיד את הטקסט המילולי המדויק)
→ שער מבני (אין ID מומצא, אין CONFIDENCE HIGH)
→ ציון groundedness (NLI רב-לשוני) — signal, לא פסק דין
→ בדיקה אנושית (pending_human_review)
```

**תוצאה מאומתת:** שליפה 6/6, resolver 6/6, שער מבני 6/6. ציון groundedness 4/6 בכיוון נכון. 178 tests עוברים, `hermes verify` תקין.

---

## 2. ההחלטה המרכזית (buy vs build — נבדקה בפועל, לא בתיאוריה)

השאלה המקורית: **האם אפשר לקנות את שכבת ה-evidence gate במקום לבנות?**

| פתרון | תוצאה אמפירית |
|---|---|
| **HHEM-2.1-open** (Vectara, חינמי) | ❌ **נכשל על עברית.** עברית זהה→0.185, עברית סותרת→0.213 (אקראי). אנגלית→0.80 (תקין). |
| **XLM-R-large-XNLI** (חינמי, מקומי) | ✅ **עובד על עברית** למרות שעברית לא ב-XNLI: זהה→0.9993, סותרת→0.0009. |
| **מסקנה** | אי-אפשר לקנות פסק-דין groundedness מוכן לעברית. מה שצריך *לבנות* הוא השכבות הדטרמיניסטיות (שליפה/resolver/שער). NLI חינמי נותן signal. |

**תובנה ארכיטקטונית מפתח:** מודל מקומי קטן (9B) לא יודע לשחזר טקסט משפטי עברי מילה במילה (הפיל "6 שנים"→"3", יישם 25% על הענף הלא-נכון). לכן הפתרון הנכון הוא **לא** לדרוש מהמודל לתמלל — אלא: המודל מצביע על `[[ID]]`, והשכבה הדטרמיניסטית מצמידה את הטקסט המדויק.

---

## 3. רכיבים שנבנו (עם קבצים)

### 3.1 מקור + provenance
- `data/originals/D37_פקודת_מס_הכנסה_נוסח_חדש.pdf` — המקור (10.8MB, **gitignored**).
- SHA-256: `4d9d397e88b5931d843af15a78e82c064a6196795ccda80468e4bb72e376349d`, **312 עמודים**.

### 3.2 Parser מודע-עמודים (TDD)
- `services/legal_parser.py` — `build_legal_tree` מקבל records `{text, page_start, page_end}` ומפיץ טווח עמודים.
- **באג שתוקן:** heuristic של "שורה קצרה=כותרת" פירק שורות PDF עטופות לכותרת מזויפת (3,213→226 nodes). fallback כותרת חופשית נשאר רק לפסקאות DOCX, לא לשורות PDF.
- `services/document_ingestion_service.py` — `_flatten_meaningful_nodes` + `persist_ingestion_receipt` נושאים `page_start/page_end` (היה NULL מקודד קשיח).

### 3.3 Evidence units (אוצרות ידנית, דטרמיניסטיות)
- `services/evidence_unit_service.py` — `build_evidence_units(pages, source_id, source_checksum, annotations)` → span רציף עם section, עמודים, span hash. דוחה anchor לא תואם.
- `data/d37_evidence_annotations.json` — annotations אוצרים (סטטוס `machine_observed_pending_human_review`).
- `scripts/build_d37_evidence_units.py` → `results/d37_evidence_units.json`:
  - `D37-9-16A-A` — סעיף `9(16א)(א)`, עמוד 43 (פטור: 6 שנים כללי / 3 פרישה/השתלמות / פטירה / סגירת חשבון / חריג משיכה חלקית).
  - `D37-125G-D-5` — סעיף `125ג(ד)(5)`, עמוד 204 (ריבית; מנתב משיכה מוקדמת מקרן השתלמות לסעיף 121).

### 3.4 שליפה מקומית (ללא API בתשלום)
- `services/local_evidence_index.py` — vector store cosine.
- `scripts/build_d37_local_evidence_index.py` → `data/d37_local_evidence_index.json` (2 entries, 768-dim, `nomic-embed-text:v1.5` דרך Ollama מקומי).

### 3.5 מסלול תשובה (pointer → resolver → gate)
- `services/d37_evidence_answer_service.py`:
  - `build_grounded_prompt` — המודל מצטט `[[evidence_id]]` בלבד.
  - `resolve_citations` — מצמיד span מילולי מדויק.
  - `gate_grounded_answer` — דוחה: חסר ציטוט / ID לא ידוע / CONFIDENCE HIGH; מגביל confidence כל עוד לא human_verified.
- `services/local_ollama_client.py` — `build_constrained_chat_payload` (`think:false`, `num_ctx:8192`, `num_predict:400`, `temperature:0`).
- `scripts/run_d37_local_evidence_cases.py` → `results/d37_local_evidence_case_run.json`.

### 3.6 ציון groundedness (חקירת buy-vs-build)
- `services/groundedness.py` — מודל-אגנוסטי: `build_premise` + `score_answer_groundedness(predict_fn injected)`.
- `scripts/score_d37_groundedness.py` — משתמש ב-`joeddav/xlm-roberta-large-xnli` (entailment) → `results/d37_groundedness.json`.

### 3.7 מקרי מבחן וגיליון בדיקה
- `eval/d37_evidence_cases_pending_human_review.jsonl` — 6 מקרים, כולם `pending_human_review` (אפס human_verified).
- `scripts/generate_d37_review_sheet.py` → `results/d37_human_review_sheet.md` (לגיא לסמן ✅/❌/⚠️).

---

## 4. ציוני groundedness (אומתו על 6 המקרים)

| # | שאלה | נכונה? | ציון | כיוון |
|---|---|---|---|---|
| 1 | זכאות (תנאי משיכה) | ❌ "3 שנים" במקום 6 | 0.976 | ❌ הפוך |
| 2 | פטירת עובד | ✅ | 0.038 | ❌ הפוך |
| 3 | משיכה חלקית | ✅ | 0.710 | ✅ |
| 4 | בסיס מס | ❌ "25%" במקום 121 | 0.040 | ✅ |
| 5 | הפניה 125ג | ❌ | 0.012 | ✅ |
| 6 | שיעור עדכני | ✅ נמנע נכון | 0.911 | ✅ |

**מגבלת entailment חד-כיווני (answer→premise):** תופס "טענה שלא במקור", אבל מחמיץ **השמטות** (מקרה 1 השמיט תנאי 6 שנים ועדיין קיבל 0.976) ומעניש **ניסוח מחדש** (מקרה 2 הנכון קיבל 0.038). לכן — **signal, לא פסק דין.** נדרשת בדיקה אנושית לפסק הדין הסופי.

---

## 5. גוטשות טכניות (קריטי להמשך)

- **HHEM** custom code דורש `transformers<5` (גרסה 5.x שינתה `all_tied_weights_keys`→`_tied_weights_keys`). **ננעץ `transformers==4.44.2`.**
- HHEM tokenizer הוא `google/flan-t5-base`; `AutoTokenizer` על ה-config המותאם של HHEM נכשל. הדרך הנכונה: `model.predict(pairs)`.
- XLM-R fast tokenizer דורש `protobuf` מותקן.
- Ollama `qwen3.5:9b-ctx64k` רץ ב-thinking כברירת מחדל → timeout. **חובה `think:false`.**
- סביבה מותקנת: `torch 2.14.0`, `transformers 4.44.2`, `sentencepiece`, `protobuf` (ב-`.venv`).
- Python: `.venv/bin/python` (3.11.16).

---

## 6. אימות

- **178 tests** עוברים (`pytest -q`).
- `hermes verify --json` → `ok:true` (bootstrap + pytest + boot + `GET /` HTTP 200). הרצה: `PATH="$PWD/.venv/bin:$PATH" hermes verify --json`.

---

## 7. מצב git

| Commit | תוכן | סטטוס |
|---|---|---|
| `e949a81` | fix: harden challenger evidence gate | **נדחף** ל-`origin/feat/hierarchical-legal-retrieval` |
| `e6f1327` | feat(d37): vertical slice (כל הקוד החדש) | **מקומי בלבד, לא נדחף** |

gitignore הורחב: `results/`, `data/regbot.db`, `data/d37_local_evidence_index.json`, `data/originals/*.pdf` — כולם reproducible/מקומיים. ה-annotations (`data/d37_evidence_annotations.json`) **מאוצרות ידנית ולכן כן committed**.

---

## 8. איך מריצים (לפי סדר)

```bash
# 1. לבנות evidence units מהמקור + annotations
.venv/bin/python scripts/build_d37_evidence_units.py \
  --pdf data/originals/D37_פקודת_מס_הכנסה_נוסח_חדש.pdf \
  --annotations data/d37_evidence_annotations.json \
  --output results/d37_evidence_units.json

# 2. לבנות index מקומי (דורש Ollama עם nomic-embed-text:v1.5)
.venv/bin/python scripts/build_d37_local_evidence_index.py \
  --evidence results/d37_evidence_units.json \
  --output data/d37_local_evidence_index.json

# 3. להריץ את 6 המקרים (שליפה + תשובה + gate + resolver)
.venv/bin/python scripts/run_d37_local_evidence_cases.py \
  --cases eval/d37_evidence_cases_pending_human_review.jsonl \
  --index data/d37_local_evidence_index.json \
  --output results/d37_local_evidence_case_run.json

# 4. ציון groundedness (XLM-R/XNLI)
.venv/bin/python scripts/score_d37_groundedness.py \
  --evidence results/d37_evidence_units.json \
  --run results/d37_local_evidence_case_run.json \
  --output results/d37_groundedness.json

# 5. גיליון בדיקה אנושית
.venv/bin/python scripts/generate_d37_review_sheet.py
```

דרישה: Ollama רץ מקומית עם `qwen3.5:9b-ctx64k` ו-`nomic-embed-text:v1.5`.

---

## 9. מה נשאר (next steps)

1. **בדיקה אנושית** של 6 המקרים (רק גיא) — סמן בגיליון, ואז הפוך ל-`human_verified` את הנכונים. זה השער הכנה.
2. **Push של `e6f1327`** אם רוצים לשתף את הקוד.
3. **יחידת ראיה שלישית** — מיסוי תשואה על הפקדות מעל התקרה (מסלול אמיתי וניתן-לציטוט, deferred).
4. **Promotion gate** — דורש 7 מקרי tuning `human_verified`; כרגע יש 3.
5. **HHEM-2.3 / VHC** — פתרון מסחרי של Vectara (לתשומת לב, לא בשימוש — בחרנו שלא).

---

## 10. לקח לשימוש חוזר (נרשם גם ב-skill `document-rag-engineering`)

רפרנס `references/hebrew-groundedness-scoring.md`: HHEM-2.1-open נכשל על עברית; XLM-R-large-XNLI עובד; entailment הוא signal לא verdict; pointer+resolver עדיף על תמלול.
