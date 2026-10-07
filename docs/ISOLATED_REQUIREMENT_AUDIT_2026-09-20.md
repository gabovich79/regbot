# Isolated requirement audit — 2026-09-20

273 code tests passed. This change adds an evaluation tool in `scripts/requirement_audit.py`; it does not change production answers or replace the runtime verifier. The previous aggregate evaluator remains historical evidence, not a trusted score.

Each approved reference requirement is examined twice in separate model calls. The answer check receives only actual answer spans and that requirement's original reference excerpts. The retrieval check receives only retrieved source spans and those excerpts. Neither route sees the other route's output, unrelated requirements or the original broad question. Maximum concurrency is two. Reference requirements never enter runtime retrieval or generation.

Verdicts distinguish covered, partial, missing and conflict. Covered requires nonempty valid source pointers and no missing parts. Partial requires both pointers and identified gaps. Invalid/missing/duplicate pointers are rejected. Provider failures remain invalid results. The result explicitly sets claim_safety_evaluated=false, professional_approval=false and acceptance_passed=false: coverage checking is not an exhaustive unsupported-claim audit.

The first paid replay found literal-wording bias and a date check that drifted back to answering the broad user question. It also returned one partial verdict without support. That experiment remains saved and invalid; cost $0.08081720. The prompt was corrected to accept semantic equivalence and not demand verbatim repetition of reference cautions, and the broad question was removed from per-requirement payloads.

The corrected replay evaluated all 11 requirements of the saved joining answer, with 22 separate route verdicts. It completed in 17.891 seconds, cost $0.07903190, and passed structural validation. No answer or index was regenerated. Structurally valid does NOT mean semantically correct: several judgments remain debatable, including scope completeness and what can be inferred from optionality. They require calibration and review, not automatic acceptance.

The audit now makes concrete deficits visible: missing employer-notification timing in the answer, incomplete enumeration of joining-notice contents, product/form-specific optional fields, and distinction between amendment and prior-version commencement. Some are retrieval deficits, some synthesis deficits; the audit's classification is advisory and must be checked against literal spans.

Do not report either the earlier aggregate evaluator's apparent 11/11 or this audit's covered/partial counts as overall accuracy. The new tool is deliberately separate from the default diagnostic scorer until its semantic calibration is adequate. No deployment, production data modification or acceptance approval occurred.
