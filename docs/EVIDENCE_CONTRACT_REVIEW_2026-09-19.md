# Preserve source qualifications before another model diagnostic

The pilot exposed a material omission even when the relevant paragraph was in
the final context: generation dropped the qualifying-deposit limitation on
interest and profits, and the semantic judge accepted the answer. This change
prevents generation from discarding qualifications identified by extraction.
It does not establish that extraction or the judge finds every legal condition.

## Implementation

`services/evidence_contract.py` binds each extracted rule to its scope,
conditions, exceptions, and known or unknown period. Every component points to
actual evidence identifiers and retains the original excerpt's content hash.
Generation selects complete units. Answer-supplied source IDs, qualification
lists and period-known flags cannot override the bound data. Rendering appends
the bound qualifications and explicitly labels an unverified period.

The independent verification request receives original excerpts, every unit and
component, and the actual display text. It must separately assess unit
completeness, component support, scope, period and preservation of conditions.
Missing, duplicate, ill-typed or unknown verdict identifiers fail closed. A
positive generic support verdict cannot override a failed dimension. One repair
is allowed, using the same units; it cannot invent a new condition set.

This is a general evidence path, with no product-specific retrieval boost or
question-specific answer injection. Evaluation fixtures are not read by runtime
services. Unknown periods and missing units produce partial or insufficient
answers. They are not silently promoted to current legal advice.

## Reference review package

`eval/diagnostic-reference-spec.json` preserves the ten original diagnostic
questions and adds assistant-authored source-scoped requirements in five
dimensions. It is explicitly unapproved development material, not the locked
20-case acceptance set. Two questions contain scope assumptions that need
correction in answers: 2016-9-8 concerns default-fund selection; 2015-9-30 concerns
actions for employers. Their original wording remains in the diagnostic.

Build the review package without API keys or paid calls:

```powershell
python scripts/build_reference_review.py --db <isolated-copy.db> --output-dir <review-directory>
```

The builder opens the DB read-only, checks each original against the pilot
manifest, extracts nine originals afresh, and attaches 27 literal evidence
spans with page/body locations, character bounds, original version and quote
hashes. A mismatching original aborts the build. PDF spans are page-sized;
DOCX spans are whole-body text with tables in document order, not invented page
numbers. These broad locators support review; they are not fine-grained
section-recall labels. The builder writes JSON and a Hebrew review document.

All ten cases remain `release_eligible: false`, and all professional approvals
remain false. Known blockers include marked-up versions in D1/D25, an equation
missing from D34's extracted text, incomplete annex review and references to
other regulations. D37's current validity and yearly caps are not verified.
The new user-supplied source survey is tracked separately in
`SOURCE_COVERAGE_2026-09-19.md`; its prose is not imported as legal evidence.

## Validation and limits

- 208 offline tests pass, including a regression using an actual D37 excerpt.
- Tests check retention, evidence/version identity, incomplete verdicts,
  wrong-scope/date decisions, repair containment and reference separation.
- The real-excerpt regression uses a simulated extraction/verifier response.
  It proves the code retains the supplied condition; it does not prove Gemini
  will extract or assess it correctly.
- No provider calls, index activation or deployment were performed for this
  change. Prior pilot measurements are not measurements of this implementation.
- One additional model stage is introduced. Its latency, cost and quality remain
  unmeasured. Appending qualifications can create repetition; readability must
  be checked in a later bounded run.

## Next gate

Resolve the reference blockers and obtain professional review of the selected
requirements before claiming accuracy or running another paid diagnostic.
Then compare a bounded run against the frozen source-backed requirements,
including omissions and contradictions, within the existing campaign budget.
This does not replace the full 60-case development/acceptance split, three
acceptance runs, deployment smoke checks or professional release approval.
Release remains blocked.
