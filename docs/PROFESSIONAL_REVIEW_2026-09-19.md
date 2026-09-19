# Source-scoped diagnostic reference review

The ten draft diagnostic references can now be presented as a local HTML review
form with their requirements, limitations and expandable original extractions.
`scripts/render_professional_review.py` builds the form from the existing review
bundle. It makes no provider calls or DB changes. Reviewer decisions start
pending and can be exported with notes and the annotation hash. Exporting is not
professional approval by itself and never activates or releases an index.

```powershell
python scripts/render_professional_review.py --bundle <reference-bundle.json> --output <review.html> --supplement eval/transfer-supplement-2025.json
```

The form has ten cases and 42 expandable source panels (including repeated pages
across cases and the supplemental source). Offline validation checked all default
decisions and exercised JavaScript export with Hebrew notes. No acceptance or
current-law approval is inferred. Scope is explicitly historical/source-bound;
the reviewer can correct or reject any interpretation.

## Transfer source found

An additional official source was downloaded from the Capital Market Authority:
letter שה.2025-384 dated 13 February 2025, concerning a clearing-system outage.
The original is 105,361 bytes, SHA-256
`9b7376b15bd959136417eaba11c4e3b0cd81b78485c92bb2149594bd6fea68fb`.
Its single page was read and visually inspected. Provenance and extracted text
are in `eval/transfer-supplement-2025.json`.

It describes regulation 5(a)'s ten-business-day rule and the commissioner's
limited extension power, then grants a specific February 2025 extension. It is
corroborating evidence, not a replacement for D15, a verified consolidated
regulation, proof of the 2016 version, or approval of 2026 applicability. The
temporary extension must not be generalized. No corpus record was attached or
activated from this letter.

The transfer case remains incomplete for the applicable cash-transfer deadline.
Other source-scoped requirements can be reviewed without pretending this gap is
closed. Completing current-law source coverage and the locked acceptance cases
remains separate work. No fresh model diagnostic is justified solely by creating
this review form.
