# Hierarchical Challenger Ranking-Gate Implementation Plan

> **For Hermes:** Implement this plan task-by-task with strict TDD. Do not alter the production answer path, production DB, document metadata, or production embeddings until every promotion gate passes.

**Goal:** Fix the challenger’s document-candidate policy so it preserves evidence found by any retrieval channel, reaches complete required-document coverage on the frozen evaluation set, and only then proceeds to scoped section retrieval and shadow evaluation.

**Architecture:** Keep the existing three independent signals—profile lexical, profile dense, and raw legal-node lexical—but make candidate generation a **union**, not an accidental winner-takes-all RRF cut. Use RRF only to order the candidate union. A document surfaced strongly by one channel must remain eligible for the section stage even if broad nodes from other documents dominate another channel. The final answer path remains untouched.

**Tech Stack:** Python 3.11, current pure-Python `DocumentRetriever` / `SectionRetriever`, OpenAI 1,024-dimension challenger embeddings, NumPy cosine similarity, JSONL evaluation fixtures, pytest.

---

## Current measured state (frozen)

Source: `results/challenger_hybrid_metrics.json`, built locally on the Mac on 2026-08-31.

```text
36 profiles
6,669 legal nodes
0 nodes missing embeddings
1,024 embedding dimensions
```

| Metric | Lexical challenger | Current hybrid challenger |
|---|---:|---:|
| Mean required-document Recall@3 | 0.714 | 0.786 |
| Mean required-document Recall@5 | 0.857 | 0.929 |
| All required docs @3 | — | 5/7 |
| All required docs @5 | — | 6/7 |

The single document-coverage failure is intentionally treated as blocking:

```text
Case: transfer-funds-multisource
Required: documents 15 + 22
Current fused top 5: 24, 15, 28, 20, 13
Dense top 5: 22, 28, 15, 24, 5
```

**Diagnosis:** The dense channel finds 22, but broad node-lexical candidates dominate RRF and remove it from the top-5 candidate set. This is a generic fusion-policy defect—not a document-22 rule, metadata edit, or query-specific boost.

---

## Non-negotiable constraints

- No query/document-specific boosts, rule tables, or special case for documents 15/22/18/36.
- A document profile is retrieval/routing material only; final legal evidence stays source-derived node text.
- No production deploy, feature flag, reindex, metadata update, or answer-path change in this plan.
- Use the existing local cache at `results/challenger_embeddings_cache_local.json`; do not rebuild it unless the node/profile construction changes.
- Keep the current 7 cases frozen. New test cases may be added only as held-out regression coverage, never used to tune a single query by hand.
- Record every ablation’s configuration and metrics. “It sounds better” is not a metric. Shocking, I know.

---

## Promotion criteria for this phase

The challenger may advance to a **read-only shadow retrieval** phase only when all conditions hold:

1. `all_required_documents_recall_at_5 == 1.0` on all 7 frozen hierarchical cases.
2. `all_required_documents_recall_at_3 >= 0.857` (at least 6/7).
3. No regression in the now-passing section cases:
   - `member-rights-section-25` retrieves document 18;
   - `loan-training-fund-section-8d` retrieves document 36.
4. The multi-source transfer case retrieves both 15 and 22.
5. Exact/missing-section case reaches its expected document but the downstream evidence gate still abstains where the requested explicit section is absent.
6. The 17 existing retrieval cases are rerun and not materially worse than the recorded flat baseline.
7. Full suite, `compileall`, `git diff --check`, and `hermes verify --json` pass.

If any condition fails, stop at challenger evaluation; do not enable shadow answer generation.

---

## Task 1: Freeze an ablation baseline

**Objective:** Make future ranking comparisons reproducible before changing scoring behavior.

**Files:**
- Create: `results/challenger_ablation_baseline_2026-08-31.json`
- Modify: none

**Step 1: Capture the exact current configuration**

Record:

```json
{
  "embedding_dimensions": 1024,
  "channels": ["profile_lexical", "profile_dense", "node_lexical"],
  "fusion": "RRF",
  "rrf_k": 60,
  "node_channel_weight": 2,
  "candidate_cutoff": 5
}
```

Record the current metrics and per-case selected documents from `results/challenger_hybrid_metrics.json`.

**Step 2: Verify it is a read-only artifact**

Run:

```bash
PYTHONPATH=/tmp/regbot-final /tmp/regbot-clean/.venv/bin/python \
  scripts/measure_challenger_hybrid.py
```

Expected: it recreates the same current metrics when pointed at the local cache; no code/database mutation.

**Step 3: Commit**

```bash
git add results/challenger_ablation_baseline_2026-08-31.json
git commit -m "docs: freeze hybrid ablation baseline"
```

---

## Task 2: Write candidate-union regression tests (RED)

**Objective:** Specify the generic retrieval contract before changing fusion.

**Files:**
- Modify: `tests/test_challenger_hybrid.py`
- Modify: `scripts/measure_challenger_hybrid.py`

**Step 1: Add a dense-only preservation test**

Create a minimal fixture where:

- document A is rank 1 in profile dense;
- unrelated documents rank high in node lexical;
- document A is absent from node lexical;
- candidate cutoff is 5.

Assert A remains in the candidate set and can appear in the final top-5.

**Step 2: Add a node-only preservation test**

Use a fixture where only a specific legal node discovers document 18-like evidence. Assert the document survives candidate formation even if profile signals are weak.

**Step 3: Add an all-required multi-source test**

Use two required documents from different channels. Assert candidate selection does not discard either before section retrieval.

**Step 4: Verify RED**

Run:

```bash
PYTHONPATH=/tmp/regbot-final /tmp/regbot-clean/.venv/bin/python \
  -m pytest -q tests/test_challenger_hybrid.py
```

Expected: new tests fail because the current RRF cutoff can discard a strong single-channel candidate.

---

## Task 3: Separate candidate recall from candidate ordering (GREEN)

**Objective:** Make RRF rank a preserved candidate union rather than decide recall by itself.

**Files:**
- Modify: `scripts/measure_challenger_hybrid.py`
- Test: `tests/test_challenger_hybrid.py`

**Step 1: Produce independent per-channel rankings**

Retain the existing channels:

```text
profile lexical rank
profile dense rank
node lexical rank, aggregated as strongest node evidence per document
```

Do not add a query-specific path.

**Step 2: Form a recall-oriented union**

Define configurable per-channel candidate depths, for example:

```text
profile lexical: top 5
profile dense: top 5
node lexical: top 5
union → deduplicated candidate documents
```

Candidate depth must be recorded in output and ablation metadata. The final document context budget stays independent of this broader recall set.

**Step 3: Order only the union**

Apply RRF across ranks available for each union candidate. Missing from a channel means no contribution from that channel—not an invented tail rank.

**Step 4: Keep diagnostics**

For each case, output:

```json
{
  "candidate_union": ["..."],
  "profile_lexical_top_n": ["..."],
  "profile_dense_top_n": ["..."],
  "node_lexical_top_n": ["..."],
  "fused_top_5": ["..."]
}
```

**Step 5: Verify GREEN**

Run the targeted tests from Task 2. Expected: all pass.

**Step 6: Commit**

```bash
git add scripts/measure_challenger_hybrid.py tests/test_challenger_hybrid.py
git commit -m "fix: preserve union of hierarchical retrieval candidates"
```

---

## Task 4: Control broad-node domination without hard-coded boosts

**Objective:** Ensure a generic document-level node score expresses evidence specificity rather than document breadth.

**Files:**
- Modify: `scripts/measure_challenger_hybrid.py`
- Test: `tests/test_challenger_hybrid.py`

**Step 1: Write a failing broad-node test**

Create a fixture with:

- one document containing many broad nodes sharing a generic token;
- another document with one short node matching a precise section/title term;
- a query containing both generic and precise terms.

Assert the broad document cannot win solely because it has more weakly overlapping nodes.

**Step 2: Implement score normalization**

Evaluate one deterministic, generic policy at a time, such as:

```text
node score = max normalized node-overlap score per document
```

Use only the strongest node, not sum/count across all nodes. Normalize by query-term coverage and/or node length so a broad body node does not receive a free advantage for being large.

Do **not** special-case Hebrew keywords, document IDs, or the transfer query.

**Step 3: Verify GREEN**

```bash
PYTHONPATH=/tmp/regbot-final /tmp/regbot-clean/.venv/bin/python \
  -m pytest -q tests/test_challenger_hybrid.py
```

**Step 4: Commit**

```bash
git add scripts/measure_challenger_hybrid.py tests/test_challenger_hybrid.py
git commit -m "fix: normalize node evidence for document candidates"
```

---

## Task 5: Run a bounded ablation matrix

**Objective:** Choose a fusion policy from measured results, not intuition.

**Files:**
- Create: `scripts/run_challenger_ablations.py`
- Create: `results/challenger_ablation_results.json`
- Create: `results/CHALLENGER_ABLATION_REVIEW.md`
- Test: `tests/test_challenger_ablations.py`

**Step 1: Write a failing test for deterministic matrix output**

Fixture: fixed miniature profiles/nodes/cases.

Assert the script emits one row per named configuration and preserves the case IDs/config fields.

**Step 2: Implement only these experiments**

| ID | Candidate policy | Fusion policy |
|---|---|---|
| A | Current RRF cutoff | Current RRF (baseline) |
| B | Union top-5 each channel | Equal RRF weights |
| C | Union top-5 each channel | Node score normalized; equal RRF weights |
| D | Union top-8 each channel | Node score normalized; equal RRF weights |

No more than these four until results justify another experiment.

**Step 3: Measure each configuration**

Use the existing local cache; do not trigger a new embedding build.

For each config report:

```text
mean document recall@3/@5
all-required-document recall@3/@5
per-case selected docs
per-case candidate union
latency
```

**Step 4: Choose a winner mechanically**

Order by:

1. all-required-document recall@5;
2. all-required-document recall@3;
3. no regression for documents 18 and 36;
4. smallest candidate union / lowest latency.

**Step 5: Commit artifacts**

```bash
git add scripts/run_challenger_ablations.py tests/test_challenger_ablations.py \
  results/challenger_ablation_results.json results/CHALLENGER_ABLATION_REVIEW.md
git commit -m "eval: compare hierarchical candidate fusion policies"
```

---

## Task 6: Add held-out legal retrieval coverage

**Objective:** Avoid tuning exclusively to seven familiar questions.

**Files:**
- Modify: `eval/hierarchical_cases.jsonl`
- Modify: `tests/test_challenger_hybrid.py`
- Create: `results/CHALLENGER_HELDOUT_REVIEW.md`

**Step 1: Add 3–5 cases not used to select fusion weights**

Must cover:

- an exact official-number or exact section request;
- a paraphrased legal condition;
- a valid multi-source question with two required documents;
- an invalid/missing-section question that must still route to the right document before the later evidence gate abstains.

Each case must name expected documents/sections and plausible distractors from the real corpus.

**Step 2: Separate tuning vs held-out reporting**

Mark the existing seven as `tuning` only for this isolated retrieval experiment. New cases are `held_out`; do not alter thresholds based on their results.

**Step 3: Verify evaluation fixture integrity**

Run the fixture tests and confirm document IDs all exist in `eval/production_corpus_manifest_2026-08-29.json`.

---

## Task 7: Re-run the full challenger gate locally

**Objective:** Produce the decision artifact for moving to section-level and shadow work.

**Files:**
- Modify: `results/CHALLENGER_HYBRID_LOCAL_RESULTS.md`
- Modify: `results/challenger_hybrid_metrics.json`
- Create: `results/challenger_promotion_gate.json`

**Step 1: Run the selected candidate configuration with the existing cache**

```bash
cd /tmp/regbot-final
CHALLENGER_CACHE_PATH=results/challenger_embeddings_cache_local.json \
  /tmp/regbot-clean/.venv/bin/python scripts/measure_challenger_hybrid.py
```

**Step 2: Run old and new retrieval cases**

```bash
PYTHONPATH=/tmp/regbot-final /tmp/regbot-clean/.venv/bin/python \
  scripts/measure_challenger_metrics.py
```

Run the existing retrieval evaluation script for the 17 legacy cases as well. Record both results; do not treat a new metric as a substitute for the old one.

**Step 3: Emit a machine-readable gate**

Example:

```json
{
  "promotion": false,
  "reasons": [
    "all_required_documents_recall_at_5 below 1.0"
  ],
  "metrics": {}
}
```

Only emit `promotion: true` when all criteria in this plan pass.

**Step 4: Full verification**

```bash
PYTHONPATH=/tmp/regbot-final /tmp/regbot-clean/.venv/bin/python -m pytest -q
/tmp/regbot-clean/.venv/bin/python -m compileall -q main.py models services scripts tests
git diff --check
env -u OPENAI_API_KEY hermes verify --json
```

**Step 5: Commit**

```bash
git add results/ eval/ scripts/ services/ tests/
git commit -m "eval: gate hierarchical challenger for shadow retrieval"
```

---

## Task 8: Read-only section retrieval and evidence-gate shadow (only after Task 7 passes)

**Objective:** Verify that a correct candidate document yields the exact legal span and enough evidence to support a reasoned answer.

**Files likely to change:**
- `services/section_retriever.py`
- `services/legal_parser.py`
- `services/claude_service.py` (only a separate challenger adapter; do not replace production path)
- `scripts/run_challenger_retrieval.py`
- `eval/hierarchical_cases.jsonl`
- `tests/test_section_retriever.py`
- `tests/test_evidence_gate.py` (create)

**Required checks:**

1. retrieve nodes only inside the chosen candidate union;
2. preserve exact section evidence for section 25 and 8(d);
3. expand to the minimum parent context needed;
4. reject a response if a required fact lacks a node/span citation;
5. for the missing-section test, route to document 18 but abstain because section 999 has no evidence;
6. build claim-bound evidence packets; no profile summary may appear as cited evidence.

**Shadow gate:** compare challenger retrieval/evidence packets with flat retrieval on the same questions. No user-visible answer change.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfitting to document 22 | No document-ID boosts; tune only generic candidate policies; add held-out cases. |
| Broad documents dominate node search | Use maximum normalized evidence score per document, not node counts/sums. |
| Candidate union increases context/latency | Union is recall-only; final section retrieval remains scoped, top-K, and measured. |
| Rebuilding embeddings causes operational churn | Reuse the completed local cache; do not build on Render production. |
| Cache ambiguity from duplicate node text | Count persisted nodes separately from unique embedding hashes; test both. |
| Accidental production migration | No production endpoint/config/DB files change before the explicit promotion artifact says `true`. |

## Explicitly out of scope

- Production rollout.
- Render embedding builds.
- Changing document 25 metadata.
- Adding a vector database.
- Generation prompt rewrite or answer-policy change.
- Hand-written legal question rules or per-document boosts.
