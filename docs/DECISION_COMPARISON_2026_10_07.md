# Frozen decision experiment — 2026-10-07

This is an offline diagnostic, not a release candidate or acceptance result.
The public pipeline is unchanged. Baseline commit: `42b52e4`.

## Inputs and reproducibility

The local run directory is `work/regbot-decision-20261007-v2` outside the repository.
It contains a SQLite backup with source asset hashes, a verified restored runtime
copy, a Git archive of the baseline, runtime file hashes, runner copy, ten questions,
reference expectations, the existing professional review receipt, and result traces.
Original source bytes match all 39 reference items. Re-extracted PDF pages match
the reference text. DOCX extraction differs for documents 22, 25, 26 and 34; the
full diffs are preserved and refreshed text is used by the oracle. The original
annotations and receipt remain unchanged; they do not approve the refreshed
extraction or any generated answer. The incomplete first preparation directory
is retained as a failed preparation, not used for inference.

## Arms fixed before execution

| Arm | Retrieval and generation | Context ceiling |
|---|---|---|
| current | Existing question planning, hybrid search, navigation, reranking, coverage, extraction, generation and verification | 24,000 tokens |
| simple | Existing deterministic hybrid RRF; top 12 hits, existing parent/neighbor expansion; one direct synthesis call | 24,000 tokens |
| oracle | Same direct synthesis prompt, supplied original pages / complete DOCX from reference source selection | 64,000 tokens |

Oracle is an upper-bound source-availability control, not a comparable speed test.
Its context is not a professionally verified minimal evidence packet. It may
include irrelevant neighboring material and needs extraction-diff review.
Simple versus current changes multiple processing stages and cannot attribute
an improvement to one stage. One run per arm is not a stability estimate.

All arms use Gemini 2.5 Flash with temperature zero and no thinking budget;
retrieval uses the existing text-embedding-3-large vectors. No web evidence.
Calls are sequential, each request capped at 90 seconds with a $0.50 reservation
limit and the persistent $20 cumulative development ledger. Errors are recorded
without automatic retries. A provider failure is not a semantic failure score.

Reference requirements are never sent to generation: only the question and
explicitly allowlisted source fields. The runner checks frozen runtime/input
hashes before execution. It does not activate or approve index versions.

## Evaluation and decision

Engine `supported` is not correctness. Direct answers are deliberately labelled
ungraded. Exact citation checks identify provenance mismatches only, not entailment;
whitespace-only mismatches must not be reported as invented legal claims.

Review against each requirement and original source must separate missing source,
lost retrieval evidence, missing qualification, unsupported claim, incorrect date,
unresolved applicability, and operational failure. Keep source-scoped correctness
separate from current-law validity. The ten cases are known development diagnostics,
mostly scoped to named circulars, not fresh generalization or locked acceptance.

Only choose a production direction after content review. A fast incomplete direct
answer does not replace the current pipeline. If both direct arms fail content,
fix the source-to-answer contract before investing in additional search complexity.
The original 40/20 acceptance split and release thresholds are unchanged.

## Commands

Run from the repository using the existing virtual environment. `prepare` needs
explicit data, manifest, references, receipt, prices and new output directory.
`run` needs that output and the persistent ledger, with provider keys supplied
through environment variables. Never put credentials into arguments or reports.

```text
python -m scripts.decision_comparison prepare --data DATA --manifest MANIFEST --references REFERENCES --receipt RECEIPT --prices PRICES --output NEW_DIRECTORY
python -m scripts.decision_comparison run --output NEW_DIRECTORY --ledger EXISTING_LEDGER
python -m scripts.render_decision_comparison NEW_DIRECTORY REVIEW.html
```

Pricing was rechecked on 2026-10-07: Gemini 2.5 Flash text input $0.30 and output
$2.50 per million tokens ([Google pricing](https://ai.google.dev/gemini-api/docs/pricing));
text-embedding-3-large $0.13 per million input tokens
([OpenAI model pricing](https://developers.openai.com/api/docs/models/text-embedding-3-large)).
No grounding calls are made in this experiment.

## Observed results and decision

All 30 scheduled attempts completed, with no retries. Operational completion is
not a correctness grade:

| Arm | Answers / attempts | Median successful latency | Cost including unresolved reservations |
|---|---:|---:|---:|
| current | 8 / 10 | 44.43 s | $0.46108831 |
| simple | 9 / 10 | 12.27 s | $0.13515544 |
| oracle | 9 / 10 | 15.17 s | $0.13088520 |

Total experiment: $0.72712895. Persistent campaign total: $8.68120178 of $20,
including $0.92940541 unresolved reservations. Current failed in evidence-unit
extraction for join and employer. Simple failed for employer and oracle for join.
These were provider ServerError responses, not proof of incorrect legal reasoning.

Targeted source review establishes these falsifying observations:

1. **Retrieval loss:** withdrawal's employee rule (D37 C137) was candidate 22 in
   simple, but omitted from its final context budget. Simple then said employee
   rules were unavailable. Oracle supplied those rules from pages 43–44.
2. **Temporal qualification:** oracle withdrawal reproduced the archived 14,140
   amount without establishing an applicable year or warning that current value
   was unverified. Its missing-information list was empty. The number exists in
   the source; the failure is applicability, not inventing the number.
3. **Synthesis omission despite available source:** oracle transfer omitted the
   15-business-day confirmations under section 6, although the complete DOCX was
   supplied, and reported no missing information.
4. **Synthesis omission despite available source:** oracle employer omitted the
   conditional reporting exemption for up to five employees under section 3(c).
5. **Lost core coverage:** simple annual-cost described direct expenses but omitted
   management fees from its answer's components; oracle included both fees and
   direct expenses. The relevant formula was absent from simple's final context.

Exact source excerpts, traces and observations are saved in the local review
bundle; a compact, non-acceptance record is committed under
`eval/results/decision_comparison_2026_10_07.json`. This is a targeted failure
review, not an exhaustive grade of all answers. Exact citation mismatches also
include whitespace changes and must not all be labelled hallucinations.

**Decision:** do not deploy either experimental direct route, delete the database,
or infer that reindexing alone fixes the product. Preserve the existing engine
as baseline. The next implementation priority is the source-to-answer contract:
identify and retain the governing scope, exceptions and temporal provisions, and
block unsupported applicability/numeric claims rather than accepting model
self-verification. Pair that with evidence selection that retains complete
relevant sections instead of consuming context with competing partial hits.
Before another broad run, demonstrate the generic change against the recorded
failures and new formulations, without embedding case answers into runtime code.
No changes to the public runtime were made in this experiment.
