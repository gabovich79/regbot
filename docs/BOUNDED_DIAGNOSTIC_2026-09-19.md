# Bounded diagnostic: execution gate

The user's approval binds the ten source-scoped reference requirements to
annotation SHA-256 `3b5245c2b21e165885b412df93ce2ca617832a949223101ee6c795e84dd07fe2`.
It does not approve system answers, current legal validity, or release.

`scripts/bounded_diagnostic.py` creates a fresh DB with only the ten selected
document records and checksum-verified originals. It copies no old chunks,
conversations, active releases, or approval states. Offline `inspect` extracts
and chunks original text; `reference` rebuilds the original evidence bundle.
Neither phase calls providers. `stage` and `run` require the persistent $20
campaign ledger. References are supplied only to the separate evaluation call.
Evaluation validates every requirement ID and Boolean field before scoring;
judgments remain advisory. Actual answers are checkpointed before evaluation.

On Linux, `supervise` runs sequential subprocesses with a 192 MiB hard address
space ceiling, a CPU ceiling, wall deadlines and a 96 MiB production reserve.
Cgroup sampling is an additional early-stop mechanism, not hard isolation.
The supervisor refuses duplicate starts and never retries workers. A new
resource/worker failure stops the campaign. Case-level answer failures are saved
and the remaining cases proceed, without retrying that question. The Linux
supervisor must not be invoked on Windows; individual offline phases work there.

## Observed stop on Render

- Service limit: 536,870,912 bytes (512 MiB).
- In-use memory: 273,379,328 bytes (~261 MiB).
- Required headroom under this conservative policy: 301,989,888 bytes (288 MiB).
- Result: `stopped_insufficient_memory_headroom`, before any new provider call.
- This is our safety threshold, not proof that the workload cannot ever fit.
- Production commit: `9e672e29e043ea05503cc6ea511035f71c4529bf`.
- Campaign settled and conservatively reserved costs: $2.12701402; no new costs.
- No deployment, activation, production DB writes or remote indexing took place.

## Offline preparation completed

All ten original checksums matched. Fresh extraction produced 1,413 candidate
chunks. D25 retains the `marked_revisions_require_source_review` flag. Other
selected files had no detected extraction flags; this is not proof of complete
semantic/legal coverage. The full historical transfer regulation gap remains.
No embeddings or answer measurements were produced in this attempt.

The remaining run requires either local Gemini access or separate compute.
Do not discard the existing development cost ledger when changing environments.
Do not lower the safety margin just to obtain a run on the production container.
Do not count this ten-case development diagnostic as locked acceptance testing.
