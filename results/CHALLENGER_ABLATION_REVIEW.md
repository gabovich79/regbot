# Candidate-Fusion Ablation Review — Local Mac

**Run date:** 2026-09-01  
**Scope:** read-only local challenger evaluation using the completed 36-profile / 6,669-node cache. No production route, DB, metadata, or user-visible answer was changed.

## Tested configurations

| Configuration | Candidate depth per channel | Node RRF weight | All-required @3 | All-required @5 |
|---|---:|---:|---:|---:|
| `union-5-node-0` | 5 | 0 | 0.429 | 0.857 |
| `union-5-node-1` | 5 | 1 | 0.714 | 0.857 |
| `union-5-node-2` | 5 | 2 | 0.714 | 0.714 |
| `union-8-node-1` | 8 | 1 | **0.857** | **0.857** |

## What improved

- Candidate union prevents the prior early-loss defect: a strong document from one channel is retained for ranking.
- Reducing node RRF weight from `2` to `1` restores document **22** to the transfer multi-source case under `union-5-node-1`.
- All configurations retain the repaired exact-section retrieval cases:
  - document **18** for section 25;
  - document **36** for section 8(d).

## Why none passes the gate

The promotion requirement is `all_required_documents_recall_at_5 == 1.0` (7/7). No tested configuration reaches it.

| Configuration | Blocking case |
|---|---|
| `union-5-node-0` | `missing-explicit-section-abstention` — document 18 absent |
| `union-5-node-1` | `missing-explicit-section-abstention` — document 18 absent |
| `union-5-node-2` | transfer multi-source and missing-section abstention |
| `union-8-node-1` | transfer multi-source — document 22 absent |

## Diagnosis

This is no longer a simple global-weight question.

1. The missing-section query explicitly names the governing law, but the document profile/entity layer does not reliably map that legal title to document 18. A non-existent section must still identify the correct law first, and only then the evidence gate should abstain.
2. The transfer query requires two authorities. A single fused top-5 list lets documents supported by broad node overlaps compete with an authority identified by dense profile semantics.

Changing a single global node weight trades one failure for the other. Continuing to tune `0/1/2` would be metric theatre.

## Gate decision

```json
{
  "promotion": false,
  "reason": "No candidate-fusion configuration achieved complete required-document coverage at top 5.",
  "next_step": "Add generic catalog/entity matching and explicit multi-source candidate coverage before a new ablation."
}
```

## Recommended next experiment (not implemented)

Add a generic, source-derived **catalog/entity match channel**:

```text
query mentions a normalized law/circular title or official number
→ catalog/profile title match contributes a candidate
→ union with lexical/dense/node candidates
→ fusion ranking
```

This is not a document-ID boost. It is the document-catalog responsibility the target architecture already requires. Then add a generic candidate-diversification policy that preserves separately supported authorities before final scoped section retrieval.
