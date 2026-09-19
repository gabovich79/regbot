# Evidence pipeline v2 — implementation and release runbook

## Release status

**Not production accepted.** Local regression tests are not the 95% retrieval /
90% answer-quality acceptance benchmark. The live corpus, secrets, current
provider tariffs, source-backed gold, three acceptance runs and professional
approval are still required. Never claim the candidate 60 questions are gold.

Existing live service: `https://regbot-wly9.onrender.com`.
Implementation baseline: `9e672e29e043ea05503cc6ea511035f71c4529bf`.
Prior hierarchical branch inspected: `e949a81aa5df117e795d2fc9f54bb63b39ba3456`.
The source-derived profile and integrity services are reused from that branch.
The D37 handoff identifies local-only commit `e6f1327`, unavailable in the remote
at implementation time. Reconcile it before closing the work; do not silently
overwrite its parser/annotation fixes. Its reported 178 tests and NLI outcomes
are historical claims, not validation of this implementation.

## Architecture and boundaries

The public chat uses `evidence_pipeline`, not the old `claude_service.stream_chat`.
The old service and its tests remain available for a baseline comparison only.
Question-specific ranking, tax/loan answer injection and auto-appended citation
footers are not called by the v2 public route. The editable legacy system prompt
is displayed as inactive in administration.

Document cards and section context are retrieval hints. Exact source text remains
separate. Stable pointers identify document/version/chunk and resolve to literal
spans with hashes. Model verification remains a fallible signal, not professional
approval. Source text is never interpreted as an instruction. Unknown dates are
not inferred from circular IDs. Legacy inferred dates are preserved for audit,
but are not considered verified by v2.

The card summary uses the first 60,000 source characters plus all detected section
headings. It is explicitly a derived navigation hint, not a complete legal
summary. Direct chunk retrieval searches the entire source regardless of card
coverage. Exact cross-document links are expanded; unresolved references remain
an evaluation concern rather than invented links.

PDFs with empty pages, corrupt extraction, or detected reversed Hebrew require
recovery/OCR outside this v1 pipeline. The service does not claim that a machine
integrity check proves visual PDF/table fidelity. Source review is required.

## Environment and controlled setup

Use a separate Render staging service/disk. Do not point staging at production's
database or expose historical conversations. First identify actual `DATA_DIR`
(the checked-in Render file uses `/var/data`; older docs mention another path).

Required environment:

* `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ADMIN_PASSWORD` via Render secrets.
* `DEMO_SESSION_SECRET`: independently generated random secret, at least 32
  characters; persistent across restarts, distinct between staging/production.
* `PUBLIC_BASE_URL`: exact external HTTPS origin for the deployed service.
* `COOKIE_SECURE=1` (only use `0` for local HTTP tests).
* `PROVIDER_PRICES_JSON`: object keyed by the exact generation and embedding
  model IDs, each with **verified** `input` and `output` USD per million tokens.
  The generation model also needs `search_call_ceiling` in USD per grounded
  discovery call. No rates are prefilled from the outdated legacy constants.
* `TRUSTED_PROXY_IPS`: explicit trusted proxy addresses, only after verifying the
  Render ingress topology. Never set an arbitrary broad trust range. The app
  rejects untrusted forwarded addresses; absent configuration may conservatively
  share the IP quota across a proxy. Configure uvicorn forwarding consistently.
* `INDEX_DOCUMENT_BUDGET_USD`: maximum metered cost per indexing operation (5 by
  default). Evaluation commands require an explicit separate batch budget.

Unknown prices disable paid public requests. Per-operation conservative token
reservations are retained on ambiguous provider errors. Grounding tariffs must
have a defensible bounded per-call charge; for variable query billing, configure
an enforceable upstream cap or leave web discovery disabled. Also configure the
provider account's usage controls. Local metering cannot retroactively reverse
an unexpected provider charge. Public reservations cap accepted daily work at
$5, resetting at midnight Asia/Jerusalem; disconnected/failed calls remain charged
for known or reserved usage. Quotas are atomic in SQLite, across processes.

Retention: public demo messages and traces are pruned after 30 days, at startup
and on public requests. Run scheduled cleanup if the service will be idle for
long periods. Legacy/admin conversations are not automatically erased.

## Backup, staging and index review

Pause administrative document changes before backup. Chat writes may continue:
the snapshot uses the SQLite backup API, not a raw copy of a WAL database.

```sh
python scripts/backup_restore.py backup /var/data /secure-backups/regbot-before-v2
python scripts/backup_restore.py restore /secure-backups/regbot-before-v2 /var/staging-data
python scripts/audit_corpus.py --db /var/staging-data/regbot.db --output corpus-audit.json
python scripts/reindex_all.py
```

Backups abort on missing source files, escaping paths, changed files or checksum
mismatches. Restore only accepts a new destination and rewrites stored artifact
paths. Verify disk space for originals, backup and both indexes before copying.

Re-indexing recovers page-aware PDF text from retained originals and DOCX tables
in document order. New versions are **staged**, not automatically activated.
In `/admin`, load versions, inspect exact text against the original, resolve
quality findings and approve each version. Warning acknowledgements need an
explanatory review note. Corruption, missing originals and OCR failures cannot be
waived. Correct metadata and re-index when identity is wrong.

Verified effective dates can be supplied to
`POST /api/admin/index/versions/{version}/metadata` while the version is pending:
ISO `effective_date`, `valid_until`, `lifecycle_status`, and an exact supporting
`source_quote`. This is a curator assertion with recorded evidence, not automatic
date extraction. Inspect it before approving the version.

Select one approved version for every active document and activate the full
manifest. Old releases/chunks remain untouched. To roll back the index, select
the prior approved versions and activate that manifest. Application rollback
uses the prior code commit and original legacy chunks. Migrations are additive.

Useful protected endpoints:

* `GET /api/admin/runtime`: paths, model IDs, active release, verified-rate config,
  and old saved system instructions; no API secrets.
* `GET /api/admin/index/versions` and `GET /api/admin/index/versions/{version}`.
* `POST /api/admin/index/{version}/review`: form `accepted`, optional `note`.
* `POST /api/admin/index/activate`: JSON array of version IDs.
* `GET /api/admin/traces/{request_id}`: candidates, context, web evidence,
  resolved claims, verification attempts, timing and provider cost.
* `GET /api/admin/conversations/{id}`: administrator-only historical access.

Public `/api/conversations` and `/{id}` enforce a signed server-issued session;
supplied `session_id` cannot grant ownership. Chat checks ownership before writing
a user message. Provider traces and full corpus downloads remain administrator-only.
SSE retains `thinking`, `text`, `usage`, `done`, `error` and adds `sources` and
`request_id`. Only verified final text is emitted; there is no unchecked draft stream.

## Gold preparation and acceptance

For an offline retention diagnostic against a restored legacy database, run
`python scripts/audit_chunk_retention.py --db COPY/regbot.db --output retention.json`.
This compares eight-word windows from extracted original PDF pages with legacy
and proposed chunks. It does not measure extraction completeness, retrieval
recall, or answer correctness; corrupted extraction can still score 100%.

To stage a reviewed-source batch on an isolated copy, use
`python scripts/stage_evidence_index.py --data-dir COPY --documents 1 11 12 36 37 38 --budget BUDGET --output staged.json`.
Replace the document list with the sources actually reviewed and BUDGET with
the authorized batch limit. A shared provider gateway caps the entire batch;
results are checkpointed after each document. Nothing is activated automatically.
Missing originals are rejected before provider spending. Extraction review and
complete-corpus activation remain separate gates.

`eval/evidence_cases_v2.jsonl` contains **60 candidate cases**, split 40/20, with
unverified references and explicit missing source URLs. Existing benchmark prose
is only a seed, never accepted as legal truth. Add exact allowed primary/secondary
source URLs where missing. Keep scenario families together. Do not tune against
the acceptance split.

```sh
python scripts/evidence_acceptance.py prepare --cases eval/evidence_cases_v2.jsonl --output work/prepared.jsonl --budget 10
python scripts/evidence_acceptance.py freeze --cases work/prepared.jsonl --output work/gold-lock.json
python scripts/evidence_acceptance.py run --cases work/prepared.jsonl --lock work/gold-lock.json --split development --output work/dev.json --budget 10
python scripts/evidence_acceptance.py run --cases work/prepared.jsonl --lock work/gold-lock.json --split acceptance --output work/acceptance-1.json --budget 10
```

Amounts above are example explicit batch caps, not preauthorized production
spend. Repeat the acceptance run into separate `-2` and `-3` files. The tool stores
fetched source snapshots, checks exact quotation matches, checks references in a
separate model call, and freezes the dataset hash. An independent model judgment
is still not human verification. Unsupported references keep the gate closed.

Each case needs confirmed `required_for_retrieval` annotations for the corpus
recall measure. Required quotations are measured against the **pre-web corpus
context**, not web snippets or source names. Run `--no-web` as a diagnostic
ablation. Review model judgments and unresolved source conflicts, especially
false-positive entailment like the D37 handoff describes.

```sh
python scripts/evidence_acceptance.py gate work/acceptance-1.json work/acceptance-2.json work/acceptance-3.json
```

The gate requires identical frozen cases, 20 unique acceptance cases per run,
95% required-source recall, 90% passing answers, no critical errors, correct
missing/conflict handling and p95 <=60 seconds. It deliberately remains closed
until professional approval. `--human-approved` records that decision only after
Guy has reviewed the package; do not set it based on model output.

Each run also records a unique run ID, code commit, models and index-state
fingerprint. The gate rejects reused run reports and differing runtime versions.
Reports preserve the actual answer and frozen reference for professional review,
and summarize results separately for answers with and without web evidence.
Malformed reports, impossible retrieval counts, contradictory pass flags, and
missing/non-finite response times cannot satisfy the release gate.
Published pricing was checked in `VERIFIED_PROVIDER_PRICES.md`; selecting that
configuration still requires the deployment/account checks described above.

Prepare a review package containing original question, answer, literal supporting
spans, URLs/pages, applicable periods, missing information, model findings, and
before/after results. Only after approval: deploy the accepted release to the
public service, smoke-test owned sessions and quotas, and retain rollback assets.

## Remaining external gates

1. Repeat snapshot checks before deployment. On 2026-09-19 the current Render
   SQLite snapshot was downloaded, checksum-verified and restored into an isolated
   local copy. The temporary transfer SSH key was removed. This does not imply
   that every original source is available or that a staging deployment passed.
2. D37 code reconciliation and original checksum are recorded in `D37_RECONCILIATION.md`; professional review of the six cases remains pending.
3. Source review, title corrections, missing originals and OCR remediation.
4. Verified current API pricing and functional provider/grounding integration.
5. Complete source-backed references, three full acceptance runs, professional
   approval and staging/public smoke tests.

Unit tests do not close these gates. No production rollout is considered complete
while any of them remains open.

## Answer verification contract

The independent verifier must return one boolean check for every candidate claim
and one explicit coverage check for every planned question aspect. A covered
aspect must reference supported claim indices. Missing or conflicting aspects
are rendered as limitations; rejected claims cannot count toward coverage.
Omitted fields, duplicate/unknown indices and malformed responses trigger the
single permitted repair attempt. If the check is still invalid, no candidate
claims are published as verified. Initial coverage gaps/conflicts are passed to
both the answer writer and verifier for reassessment against the final evidence.
This contract prevents silent omission in structured output; its semantic
accuracy still requires real provider evaluation and professional acceptance.
