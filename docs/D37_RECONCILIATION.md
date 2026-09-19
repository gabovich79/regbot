# Reconciliation with the hierarchical branch

Reviewed `c950dbfc03bf5694210186c05108b8dc2a04792e` and its D37 implementation commit `e6f132724072b5d3d485ab8a05275fabdc0210eb` on 2026-09-19.

Retained the source-anchored evidence-unit builder, curated annotations, six pending-human-review cases, builder CLI and unit tests from that branch. These are diagnostic/review fixtures; they do not inject D37-specific retrieval rankings or answers into the general pipeline. They are separate from the frozen acceptance split.

The new pipeline already preserves cross-page sections, does not treat every short PDF line as a heading, resolves model pointers to literal source spans, and marks semantic verification as an automatic check rather than professional approval. The old parser persistence patch targets different tables; it was reviewed but not transplanted over the versioned index schema.

Ollama and XNLI remain experiments on the hierarchical branch. The agreed deployment keeps Gemini and the existing embedding provider. The handoff's false-positive NLI result is a reason to retain independent completeness checks and human acceptance, not to use NLI scores as release approval.

Render inspection found the original D37 PDF at `/var/data/originals/37.pdf`: 312 pages, SHA-256 `4d9d397e88b5931d843af15a78e82c064a6196795ccda80468e4bb72e376349d`, matching the handoff. The PDF need not be committed to Git. None of the six cases has been promoted to human-verified by this reconciliation.
