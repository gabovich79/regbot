# Development pilot: release blocked

Ten diagnostic questions were run through the legacy path and a staged evidence
index, followed by two bounded correction rounds. This is not an acceptance
suite. Do not run further paid diagnostic rounds without first addressing the
engineering decision below and the user's agreed scope.

## Provenance and limits

- Production remained at `9e672e29e043ea05503cc6ea511035f71c4529bf`.
- New pipeline rounds used `16ee993`, `cfee001`, and `3413b1b` respectively.
- Offline review reporting was updated at `e650451` after the final engine run.
- Models: Gemini 2.5 Flash and text-embedding-3-large.
- Legacy search covered 36 active documents. New search covered ten selected
  original files, 1,413 staged chunks, with no review or activation promotion.
- No web supplementation or conversation history was used.
- References were frozen before answer runs, but are machine-checked diagnostic
  references, not professional gold. One failed independent reference review.
- The baseline retained legacy retrieval/prompts/postprocessing; its metered
  generation wrapper collected a non-streamed response.

## Observations

| Run | Completed answers / attempted | Technical failure | Successful-answer P95, nearest rank |
|---|---|---|---|
| Legacy | 10/10 | none | 28.5 s |
| Initial new | 9/10 | invalid rerank identifier | 71.8 s |
| Correction 1 | 9/10 | malformed/truncated JSON | 62.9 s |
| Correction 2 | 9/10 | provider 504 during answer generation | 66.6 s |

These tiny latency samples exclude the external diagnostic judge. Automatic
supported labels are not accuracy scores. The last run had seven supported and
two partial answers; neither count establishes release readiness.

Observed planner drift erased a named product and translated a first-turn query
into a generic English question. Another rewrite treated a circular identifier
as a date. Keeping the literal first-turn question restored relevant source
chunks in the inspected case. Long rerank IDs were replaced by short aliases
resolved back to immutable evidence IDs. Unverified factual gap prose is no
longer displayed. Index memory and verifier output handling were improved.

Unresolved: qualification/exception omissions can survive both generation and
verification even when present in the supplied source. In the withdrawal answer,
the employee paragraph omitted the source's qualifying-deposit limitation on
interest/profits; the independent judge did not flag it. A separate question
returned insufficient information despite its relevant clause being in context.
The final run recovered that clause in its answer, but broad reliability is not
established. Judges produced schema deviations and contradictory completeness
verdicts, so `scripts/pilot_report.py` reports review requirements, not accuracy.

Provider usage plus conservative outstanding reservations totaled $2.12701402
against the user-approved cumulative $20 development ceiling. Settled estimates
were $2.05092882; outstanding reservations were $0.0760852. This is an application
ledger, not a provider invoice. No pilot process was left running.

The isolated DB shared the production container's 512 MB memory limit. Initial
indexing hit OOM; a production worker may also have restarted. Separate code and
data do not provide operational isolation. Subsequent round checks showed no
additional OOM kills, and the production version endpoint returned HTTP 200.

## Decision before more model evaluation

1. Prepare reviewable source-backed units for rule, conditions, exceptions,
   population and effective period. Resolve reference disagreements and freeze
   the planned 40 development / 20 acceptance split without moving pilot cases
   into the held-out set.
2. Bind a generated rule to its limiting conditions in the evidence contract.
   Verify coverage against complete source units, not just citation existence.
   Demonstrate omission detection before another paid pilot.
3. Complete recovery/version/OCR review (27 of 36 originals attached in the
   recovery copy, nine unresolved; D8/D9 extraction still needs repair).
4. Use operationally separate staging for web, provider failures, concurrency,
   time limits and acceptance. Retain the original three-run and professional
   approval gates. Do not merge/deploy this draft as an accepted release.

## Offline review packet

After archiving the final output as `pilot-round2-results.json`, run:

```sh
python scripts/pilot_report.py --data-dir /path/to/isolated-pilot
```

It reads the frozen references, source receipt and archived baseline/round JSON
files and writes `pilot-summary.json` plus `pilot-review.md`, retaining every
answer and raw judge result. It makes no provider calls. Full review data remains
on the private persistent disk; no conversation DB or secrets are included in
the review archive. The summary and review packet do not approve an index.

Reference SHA-256: `db4c48813888a35af257f384af0fc47adaaf0101f724a4099f08f9712fab927f`.
Final results SHA-256: `a5fbe55b7e1aefcae8946edd0c51bcef2a8e71ebb7526001e6982e2986e05498`.
