# Hierarchical Retrieval Challenger — First Results

**Status: production plan — challenger runs read-only against local corpus export.**

## Architecture built

```text
profiles (document cards)
→ DocumentRetriever (title/topics/outline/section overlap + official number + explicit section)
→ SectionRetriever scoped to selected documents (heading + raw text overlap)
→ parent-section expansion
```

- `services/document_retriever.py`
- `services/section_retriever.py`
- `services/embeddings.py` (OpenAI-compatible embedding helper; no key used yet)
- `scripts/run_challenger_retrieval.py`
- `scripts/measure_challenger_metrics.py`
- `scripts/build_challenger_embeddings.py` (prepared, requires API key to run)

## Metrics on 7 hierarchical cases (lexical only)

| Metric | Value |
|---|---|
| Document Recall@3 | 0.714 (5/7) |
| Document Recall@5 | 0.857 (6/7) |
| Mean rank of first hit | 2.17 |

Per-case:

- age-dependent-tracks-v2 → 38 (hit@1)
- transfer-funds-multisource → 22 (hit@1)
- member-rights-section-25 → 18 (hit@1 after section-mention extraction)
- training-fund-withdrawal → 37 (hit@1)
- loan-training-fund-section-8d → miss (36 not in top-5; RTL extraction breaks `סעיף 8(ד)` string matching)
- withdrawal-and-tax-parameter-synthesis → 37 (hit@4; loses to broader profiles)
- missing-explicit-section-abstention → 18 (hit@3)

## Honest assessment

The challenger works structurally: document-first then section-with-parent
expansion is implemented and beats zero-heading flat matching on the same
corpus. But at lexical-only it is not yet better than the existing production
flat retriever end-to-end. The two hard cases (loan section 8(ד), mixed
synthesis) require dense semantics, which the lexical layer cannot provide.

The embedding cache script is ready but requires an API key; the environment
currently has no usable `OPENAI_API_KEY`. Running `build_challenger_embeddings.py`
inside the Render shell (or with a provided key) is the next measurable step.

## What this does NOT do yet

- No FTS5 query path wired (schema exists).
- No reranker.
- No evidence-gate integration.
- No shadow run against production answers.
- No generation change.

## Next gate

Run embedding build + hybrid (dense+lexical+RRF) evaluation on the same 7 cases,
then compare against flat on the 17-title cases. Decision to switch production
retrieval only after that comparison and Guy's approval.
