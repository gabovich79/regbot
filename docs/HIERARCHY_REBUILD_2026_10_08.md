# Source hierarchy and navigation rebuild — 8 October 2026

This is a development result, not an acceptance pass or a production release.

## Implemented

- PDF navigation recognizes uniquely aligned bold statutory captions, including visual RTL ordering of amendment metadata. Amendment dates alone are excluded. Plain bold captions require a following numbered clause. Text and page mappings remain unchanged.
- Explicit annex part labels split DOCX sections outside tables. Re-extraction includes the existing numbering and table fixes that were absent from older indexes.
- Oversized source sections are recorded as review issues, not silently declared complete.
- The heading menu ranks headings by the question before applying its size limit. Previously the beginning of long documents displaced relevant late sections.
- The reranking response schema requires exactly as many ratings as candidates; validation still rejects missing, duplicate or invented IDs.
- Context packing distinguishes an oversized source section from an oversized dependency bundle. A fitting selected section stays whole, while excluded dependencies remain explicit gaps.
- The rebuild tool can reuse embeddings only when the source hash, model, title, context and chunk text satisfy the relevant equality checks. A changed embedding input is sent to the metered provider again. No active index is changed.

## Corpus measurements

A consistent SQLite/source snapshot was restored to a separate local runtime. Original binary checksums were verified. Ordered section and chunk coverage was checked against the extraction, allowing only whitespace-only gaps between sections.

| Document | Old parents | Final parents | Final maximum section tokens | Remaining review issues |
|---|---:|---:|---:|---|
| 22, transfer circular | 14 | 36 | 5,338 | Unresolved source numbering |
| 25, employer reporting | 5 | 71 | 2,248 | Marked revisions |
| 37, income tax ordinance | 28 | 585 | 87,189 | Inferred hierarchy and one oversized remainder |

Document 37's extracted-text hash is unchanged. Documents 22 and 25 changed because numbering and table extraction were refreshed, with the same original binaries. All replacement versions remain pending review. The intermediate D37 version with 612 sections was superseded after date-only lines were found among its navigation boundaries. The final rebuild reused 1,288 identical vectors and embedded 67 changed inputs.

## Recorded development attempts

Three new formulations of already-known development topics were used; they are not a held-out acceptance set. Every attempt is retained. No web sources were used.

| Stage | Withdrawal | Transfer | Employer |
|---|---|---|---|
| Rebuilt sources, old menu order | Partial, 42.6 s; wrong subject emphasized | Provider ServerError, 46.2 s | Partial, 45.9 s |
| Relevant-heading menu, final source versions | Incomplete reranker output rejected, 14.3 s | Partial, 61.7 s | Partial, 46.0 s |
| Exact rating-count schema | Insufficient, 41.6 s | Not rerun | Not rerun |

The first withdrawal attempt mostly saw general income provisions. After the menu change, the employee and self-employed withdrawal sections reached reranking. The final withdrawal attempt still returned no answer: all 18 generated claims were rejected as `unscoped_numeric_parameter`, because complete evidence units bundled general rules with numeric requirements whose applicability period was unknown. This is a remaining synthesis/contract-granularity defect; do not relax the numeric-date safeguard to hide it.

Spot inspection found the conditional small-employer exemption and the 15-business-day transfer confirmation requirements in generated answers, where earlier diagnostics had omitted them. This is not a full legal grading or proof of current applicability. Marked revisions, unresolved references, omitted evidence and temporal validation still limit the answers.

After the live attempts, an additional packing fix was checked with saved ranked IDs, without provider calls. Withdrawal context increased from four to five chunks and contained both selected source sections completely (4,191 tokens); unresolved dependencies remain explicit. Transfer and employer contexts were unchanged (23,743 and 10,014 tokens). No live answer on that final packing change is claimed.

## Validation and remaining work

353 automated tests passed. Coverage includes RTL caption recognition, amendment-only rejection, exact page alignment, late-heading discovery, strict ranking counts, identical-input embedding reuse, and complete sections under oversized dependency bundles. These tests verify software behavior, not regulatory accuracy.

This increment consumed $0.55136384 including retained reservations. The persistent campaign ledger totals $9.47072310 of $20, including $1.03907171 in unsettled reservations. No ledger reset or unmetered provider calls were used.

Next: separate independently answerable rules and their actual qualifications without allowing a model to discard exceptions; preserve date requirements for financial parameters; resolve source-backed cross-document references; review remaining source issues. Then measure end-to-end completeness on distinct questions before attempting the locked acceptance set. Production and its database were not modified.

Local run artifacts: `work/regbot-hierarchy-20261007/` (outside the repository): snapshot, restored runtime, staging manifests, all seven live attempts, and packing replay. Machine-readable measurements are in `eval/results/hierarchy_rebuild_2026_10_08.json`.
