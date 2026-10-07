# Section context and date provenance — 2026-10-07

Implemented in the development branch. No production deployment or index activation.

## General runtime changes

The retrieval pipeline now packs contiguous complete stored parent sections,
rather than filling the context with independent ranked chunks before expanding
their parents. Identical headings in separate positions are not merged. It adds
identifiable governing sections (definitions, scope, commencement and transition)
and unambiguous local numbered references. Explicit external references remain
unresolved instead of silently binding a same-numbered local section.

Bundles are budgeted with their serialized metadata and never sliced to fit.
Legacy indexes sometimes treat an entire chapter as one parent. When that parent
alone cannot fit, a bounded contiguous neighborhood around ranked hits is retained
and explicitly marked partial. This is a compatibility fallback, not a claim that
the legal section is complete. Such fallback, omitted sections and unresolved
references prevent the answer from receiving a fully supported status. Coverage
also receives these limitations. Resolving external references and repairing the
oversized stored hierarchy remain unfinished work.

A separate deterministic guard now checks whether structured applicability dates
actually occur as calendar dates in the cited source text. A year-first circular
identifier is not calendar evidence. Unsupported period components are removed
before synthesis, period_known becomes false, and a gap is retained. Supported
date formats include Hebrew month names, conventional day-first numeric dates,
and ISO dates with explicit temporal cues. Unrecognized formats fail closed.
Literal occurrence is necessary but does not establish legal applicability;
semantic verification remains required. This is not a universal numeric/date
verifier and does not prove all textual claims are correct.

## Validation

- 343 regression tests passed. New cases cover preserved continuations, complete
  bundle budgeting, explicit partial fallback, repeated headings, local/external
  references, and circular identifiers mistaken for dates.
- Replayed nine saved retrievals without provider calls. Every result stayed
  within 24,000 tokens and had unique evidence IDs. Legacy partial parents were
  exposed in employer, transfer and withdrawal. Join still omitted eight section
  seeds under the budget. Replay measures structural behavior, not legal recall.
- Rebinding the saved join extraction blocked 19 period components whose claimed
  dates were not present in their cited excerpts, including the previously
  reported 2021-09-05 error. This is replay evidence, not a new full acceptance run.
- Two new formulations ran through the full modified pipeline without web:
  fees 48.484 seconds, annual-cost 46.187 seconds. Both returned partial answers.
  Their content has not received exhaustive professional correctness grading.

## Separate thinking experiment

Four new formulations of known development topics used identical fixed original
evidence, prompt, model and output ceiling in paired runs with thinking budgets
0 and 1,024. These are not unseen topics or locked acceptance cases. Six of eight
calls returned answers; employer failed with ServerError in both arms. No retries.

Targeted review did not establish a consistent winner. Both annual-cost responses
covered the main cost components and product exclusions. For loan, thinking added
the first-year-repayment exception, but omitted the aggregate 10% lending limit
that the zero-thinking answer included. No automatic correctness percentage was
assigned. Thinking therefore was not enabled by default in the public pipeline.

## Artifacts and cost

Local artifacts outside Git: `work/section-replay-final-20261007-v2.json`,
`work/regbot-thinking-20261007`, and `work/regbot-section-validation-20261007`.
Reproducible entry points are scripts/replay_section_context.py,
scripts/thinking_comparison.py and scripts/section_pipeline_validation.py.

Thinking comparison cost/reservations: $0.14290240. Full-pipeline validation:
$0.09525508. This implementation cycle total: $0.23815748. Cumulative development
ledger: $8.91935926 of $20, including $0.99316361 unresolved reservations.

The code changes are complete for this increment. Product completion remains
blocked by corpus hierarchy/dependency gaps, synthesis completeness, temporal
validity and the original professional acceptance/release gates. No inference
of broad accuracy improvement should be made from these limited checks.
