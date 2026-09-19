# Completed local diagnostic (development only)

Engine commit: 579684c (no engine edits during the measured questions).
Reference annotation SHA-256:
`3b5245c2b21e165885b412df93ce2ca617832a949223101ee6c795e84dd07fe2`.

All ten checksum-verified documents were indexed into 1,413 staged chunks.
SQLite integrity is OK; active_index has zero rows. No production deployment
or activation occurred. All ten questions ran once without web supplementation.
This is not the locked acceptance set or current-law approval.

## Results and release blockers

- Six responses contained substantive content; three were insufficient-information
  responses; one failed with a Gemini 504. Returning content is not an accuracy pass.
- Fees, annual-cost and loan failed structural validation: the generator returned
  component IDs (e.g. `U1:rule:0`) where unit IDs (`U1`) were required. The one
  internal repair repeated the same error. Retrieved sources were available.
- The separate judge reported omitted requirements in withdrawal (7/8 answered),
  joining (3/11), employer reporting (6/9), and personal eligibility (0/5).
  These are advisory results requiring review, not certified accuracy measurements.
- The loan judge incorrectly credited 7/7 requirements while the final answer
  only refused to answer. Its reasons evaluated retrieved evidence instead of
  actual answer text. This semantic evaluation failure is not caught by schema
  validation. Preserve the raw result but exclude it from accuracy credit.
- No overall accuracy or acceptance recall percentage is reported. The automatic
  65/65 retrieved-requirement result cannot establish the acceptance recall gate,
  especially given the judge failure and broad reference locators.
- Personal eligibility did not request the missing product/personal circumstances.
- No unknown chunk IDs were found in rendered citations. This does not prove
  semantic support, completeness, or appropriate temporal applicability.
- Nearest-rank p95 over ten answer attempts was 67.47 seconds, above the 60-second
  goal. This tiny sample is diagnostic, not production latency certification.

## Execution and spend

Local TLS was configured to trust the Windows certificate store without bypassing
verification. An OpenAI environment value accidentally containing an entire setx
command was corrected and model access verified without exposing credentials.
After six documents, Python crashed in ntdll.dll with 0xc000070a. Offline D34
extraction and DB integrity checks passed. The user authorized continuation; the
six completed versions were verified and skipped. Remaining indexing and all ten
questions then completed. The original native crash root cause remains unknown.

The local ledger carries forward the prior Render total of $2.12701402, including
unresolved conservative reservations. There were no parallel remote development
jobs. The cumulative campaign total is $3.23531241 against the $20 ceiling; the
local portion is $1.10829839 including reservations. Keep this ledger when resuming
development; do not restart from the older Render balance.

## Next bounded implementation scope

1. Enforce explicit unit-ID versus component-ID schemas in generation and repair;
   do not silently map component IDs to units and lose their qualifications.
2. Trace required question aspects through evidence-unit extraction, answer and
   verification, and distinguish missing user facts from missing source evidence.
3. Require each credited evaluation requirement to quote a literal span of the
   actual final answer; assess retrieval separately and flag refusal/credit conflicts.
4. Handle provider timeout within the overall deadline and reserved cost ceiling.

No new answer run or engine patch was performed after measuring these failures.
Deleting the DB is not a remedy for these identified contract/evaluation failures.
