# Challenger Hybrid — Local Mac Results

**Run date:** 2026-08-31  
**Scope:** read-only challenger evaluation; no production retrieval or database path was changed.

## Build verification

| Check | Result |
|---|---:|
| Document profiles | 36 |
| Legal retrieval nodes | 6,669 |
| Nodes missing an embedding | 0 |
| Embedding dimension | 1,024 |
| Duplicate-content nodes reusing an existing vector | 97 |

`embedded_nodes: 6572` in the runner output counts unique text hashes, not persisted node records. The cache contains vectors on all 6,669 nodes.

## Results — 7 hierarchical cases

| Metric | Lexical baseline | Local hybrid | Delta |
|---|---:|---:|---:|
| Mean required-document recall@3 | 0.714 | **0.786** | +0.072 |
| Mean required-document recall@5 | 0.857 | **0.929** | +0.072 |
| All required documents present @3 | — | 0.714 (5/7) | — |
| All required documents present @5 | — | 0.857 (6/7) | — |

### Per-case result

| Case | Required docs | Hybrid @5 | Result |
|---|---|---|---|
| Age-dependent tracks | 38 | 38 #1 | Pass |
| Fund transfer, multi-source | 15 + 22 | 15 #2; **22 absent** | Fail (coverage) |
| Member rights, section 25 | 18 | 18 #3 | Pass |
| Training-fund withdrawal | 37 | 37 #1 | Pass |
| Training-fund loan, section 8(d) | 36 | 36 #2 | Pass |
| Withdrawal + tax-parameter synthesis | 37 | 37 #1 | Pass |
| Missing explicit section / abstention | 18 | 18 #4 | Pass for document recall |

## Interpretation

This validates the core hierarchical approach:

- the rebuilt node corpus fixes the prior false absence of documents **18** and **36**;
- the difficult `section 25` and `section 8(d)` cases now retrieve their required document;
- hybrid retrieval improves mean document recall over the lexical-only challenger.

However, this is **not a promotion result**. It misses document **22** in the one required multi-source transfer case. The diagnostics show dense ranking places 22 at #1, but node lexical candidates favor broad transfer-related nodes from other documents and RRF pushes 22 out of the top 5.

## Gate decision

**Do not enable the challenger in production or shadow answer generation yet.**

Next work must be an isolated ranking experiment with the same fixed cases: normalize/limit broad node candidate influence or make its RRF weight confidence-sensitive, then remeasure. Retain a change only if it reaches 7/7 all-required-document coverage at the chosen K without regressing section 25 or 8(d).
