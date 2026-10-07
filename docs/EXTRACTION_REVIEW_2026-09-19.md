# Equations and amendment markings in source evidence

The source review found two additional extraction failures. Word equations were
absent from paragraph text, including D34's annual-cost formula. D25's replaced
commencement date appeared beside the replacement as if both were ordinary
operative text. These are source representation failures, not embedding issues.

## Changes and verification

DOCX body and table extraction now preserves supported OMML equations in a
labelled linear representation. Fractions retain numerator/denominator grouping.
Unknown equation structures raise a review error instead of silently flattening
or omitting mathematical content. The parser intentionally supports only the
structures exercised here; it is not a complete Word math interpreter.

Explicit run strikethroughs and tracked deletions/insertions are labelled in
extracted text. The extractor does not accept amendments or infer effective
dates. Marked revisions add a source-review issue and make an otherwise-current
derived card's lifecycle unknown. Existing activation review remains required.
Inherited style-based markings and image equations are not comprehensively
handled; this is not a universal lossless-DOCX claim.

All nine available DOCX originals in the local recovery copy were extracted
successfully. D34 now includes six equations. Deletions were surfaced in D25,
D33 and D35. The latter two are additional review work, not automatically
accepted clean texts. Source bytes, stored legacy extractions and indexes were
not overwritten. Future re-ingestion must build new versions.

D34's XML contains the annual-cost sum De + Fc + Fa, AD = (D1 + D2) / 2,
and Fc = d / 20. Those representations match the explanatory prose and worked
example. D25's XML explicitly strikes the older 1 February 2020 wording and
underlines the replacement group/date. A DOCX visual render was attempted but
LibreOffice was unavailable; the DOCX conclusions rely on original OOXML
structure, not a claimed visual inspection.

D1 PDF pages 1, 8, 9, 11, 16 and 20 were visually inspected. The 2021 amendment's
commencement and the underlined direct-expense additions were distinguished
from earlier commencement clauses. The starred form fields differ by product;
the reference does not make every field mandatory. Remaining annex A–C text
was reviewed from extraction. No assertion of current-law validity is made.

## Reference package

The ten diagnostic cases now link to 39 extracted spans from nine checksum-bound
originals. Evidence IDs include an extraction hash as well as original hash and
location, so a repaired extraction cannot reuse an obsolete text identity.
The package records technical resolutions separately from professional review.
All cases remain unapproved and ineligible for acceptance scoring.

The transfer-regulation candidate previously downloaded for D15 was inspected.
It differs from the legacy text and has an unfilled signature date. It was not
attached as D15 or used to settle the cash-transfer deadline. The reference
explicitly leaves regulation 5(a)'s applicable deadline unresolved. This gap
must not be closed by asking the professional reviewer to approve an invented
deadline or by substituting the circular's request-transmission deadline.

Validation: 216 offline tests passed. No model charges, index activation,
production DB changes or deployment occurred. These tests are software and
provenance checks, not a regulatory accuracy percentage.

## Rebuilding the index

Re-extraction and a new index are appropriate after source review because a new
embedding cannot recover a formula absent from its input. Use an isolated copy,
immutable originals and the staging/review/release path. Do not delete the whole
DB: conversations, settings, source provenance, spending records and rollback
state are separate from the derived index. A rebuild still needs retrieval and
answer verification; it does not by itself fix synthesis omissions or establish
current legal validity. No destructive rebuild has been authorized or performed.
