# Evidence units, source hierarchy and provenance — 8 October 2026

These are development diagnostics, not acceptance results. Production and the active index were not changed. No benchmark-specific answer or ranking rule was added.

## Changes and measured effects

- Extraction now separates independently assessable legal consequences, while retaining every governing qualification and any financial dependency. In the withdrawal diagnostic, previously bundled unknown-period numeric parameters blocked the whole answer. The new extraction yielded 19 units and 17 accepted claims in 50.672 seconds. Three numeric candidates remained blocked; the answer remained partial. These counts are not a professional correctness score.
- DOCX source typography distinguishes root numbered headings from nested list numbering. Paragraphs are aligned sequentially against the exact extracted stream, including repeated text and table cells; mismatched source text cannot supply hierarchy. Document 25 changed from 71 parents/81 chunks to 22 parents/39 chunks with identical source text and original checksum. A previously ambiguous local reference to section 3 resolved. Typography is still an inference requiring review; marked revisions remain unresolved.
- A development case attached a 2021 date from document 1 to rules from document 38. The semantic verifier had accepted it. Deterministic date binding now requires the cited date to occur in the same source document/version as each rule, unless a separately validated governing relationship is implemented. Pointer laundering with a date-free rule excerpt does not pass. Replay reduced known periods from nine to four; a new live run was partial in 36.422 seconds. This does not validate cross-document scope, nor prove legal applicability merely from a literal date.
- Provider failures now retain safe status/code fields. The KYC development case failed with 504 DEADLINE_EXCEEDED during unit extraction. No hidden retries were added and no failure was relabelled as success.
- An optional, disabled-by-default composition path uses extracted rules verbatim, retaining their qualifications and the normal resolver/verifier. Fixed-extraction replays of employer and transfer cases needed one verification call; withdrawal still needed bounded repair and verification. Replay times are not end-to-end latency. This experiment has not been promoted to the default.
- PDF extraction records detected ruled-table regions and flags them for structure review. Detection failure is also explicit. It does not reconstruct table relationships, detect every possible unruled table, or approve layout from an empty detection result.

## Current official source recovery

Downloaded official employer-deposit circulars 2025-9-3, 2025-9-6 and 2026-9-2, preserving original checksums and retrieval receipts. The March 2026 first page identifies circular 2026-9-2 dated 26 March 2026 even though its URL filename contains `draft`. Classification must use the original content, not the filename.

Visual comparison of page 19 of the 2026 PDF with flat extraction found lost column-to-cell relationships. Automated table detection flags pages 14, 18 and 19; that list is not exhaustive visual certification. These files are candidate sources only, not approved or activated. Their amendment relationships and section-specific commencement must be reviewed; the appendix contains older document dates. Recovery does not establish that no later amendment exists.

Receipts and findings: `eval/results/source_refresh_2026_10_08.json`. Local originals and rendered inspections are retained in `work/regbot-source-refresh-20261008` alongside the checkout, not committed as binaries.

## Verification and remaining gates

366 local tests passed. Machine-readable development measurements are in `eval/results/unit_granularity_2026_10_08.json`. Some live runs exceeded the 60-second target or failed at the provider. No aggregate 90%/95% acceptance claim is available.

Development budget charged or reserved: $10.01047898 of $20, including $1.12449431 of unsettled reservations. Fixed replays, source retrieval and local tests must not be described as fresh end-to-end acceptance runs.

Next gates: preserve table relationships and review amendment/version links; resolve cross-document applicability and scope from explicit sources; validate the simpler composition path on additional development cases before changing defaults; complete independently reviewed references for the 40/20 split. Production release still requires the original acceptance thresholds, three full runs and professional approval.
