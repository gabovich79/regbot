# Navigation and explicit candidate ratings — 2026-09-20

Code: 60a2ce5. 263 tests passed. No production changes or index activation.

Two general fixes:

1. Navigation stores document titles once and preserves original section headings without repeated source snippets. The exact serialized document/section menu stays within 8,000 tokens. The saved diagnostic now exposes all 76 headings with zero omitted, versus 35 exposed and 41 omitted previously. Up to 20 section pointers are selected; this does not promise exhaustive final evidence.
2. Reranking now requires a score (0–3) and reason for every candidate. Missing, duplicate, unknown and malformed entries fail validation. Code sorts by the supplied score, using prior retrieval order only for ties. Zero-score candidates are excluded. No question-specific rules or legal answers are embedded.

Measured retrieval on the same isolated 97-chunk document version and saved question plan: 40 valid candidate ratings; final context 35 chunks / 23,931 tokens. The post-joining notice provision (source chunk C40) moved from omitted to second selected source, with score 3 and its continuation in final context. Rejection, account consolidation and beneficiary provisions also appear. This verifies inclusion of identified passages, not 95% overall evidence recall.

Provider experiments retained locally:

| Experiment | Result | Cost |
|---|---|---:|
| Compact menu retrieval | 76 headings, zero omitted | $0.00738659 |
| Downstream answer on compact retrieval | Gemini 504, failed, no score; 31.765 s | $0.05393920 committed including unsettled usage reservation |
| Explicit-rating retrieval | Complete 40-candidate ratings accepted | $0.01010369 |
| Downstream answer on rated retrieval | Partial answer, 42.75 s excluding retrieval | $0.06947540 including evaluation |

The final downstream run froze understanding and retrieval. It is not an end-to-end latency or acceptance test. Its displayed answer now includes a joining-notice deadline and account-consolidation notification/objection periods. This observation is not professional verification of every condition.

External evaluation remains invalid: three requirements were marked answered=true with no supporting answer-span IDs. In addition, manual inspection found that a narrow exception in an answer span was credited against a broader scope requirement. The runtime validator rejected the overall evaluation; none of the apparent 11/11 positive judgments should be reported as accuracy. Literal pointer validity alone does not establish semantic completeness.

Release remains blocked by incomplete answer coverage, unreliable semantic evaluation, corpus-wide structural validation and the unperformed locked acceptance runs. Next work must distinguish full requirement support from topical/partial matches and address remaining source hierarchy and context-selection gaps. No repeated run on an unchanged input was treated as a fix. The previous 504 remains recorded as a failure, with its budget reservation retained.
