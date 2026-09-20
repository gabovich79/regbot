# Section navigation diagnostic — 2026-09-20

Code: 3cd31f2. 254 local tests passed. No production deployment or index activation.

Changes:
- Added a bounded source-section navigation route alongside existing direct lexical/dense retrieval. It lists contiguous original sections from up to five discovered document versions, selects up to 12 source section pointers, merges with direct hits, and reranks at most 40 original chunks. Navigation text itself never becomes evidence.
- Trace includes the navigation catalog, omitted section count, selected original IDs, and final rerank candidate IDs. Unknown pointers are rejected. Direct search remains available.
- Fixed PDF extraction layout where the section number, period and heading are on different lines. Four-digit years followed by a period are not interpreted as numbered headings. Original source text remains unchanged.
- Reranking instructions explicitly include provisions governing scope, commencement and transitions.

A new isolated version of document 1 was staged: d73d2836503d4732a60c8bfaa86e5672, 97 chunks, $0.02249396. Original and previous staged versions were retained.

Controlled retrieval with the saved question plan returned 34 chunks, 22,266 context tokens. Operative joining, online joining, rejection, definitions, scope and commencement provisions now appear alongside forms. This is source-level observation, not proof of complete regulatory coverage. Navigation cost $0.00811919 including embeddings/reranking. The earlier navigation-only experiment before the PDF-number fix cost $0.00860849 and did not establish adequate operative coverage.

Known navigation limit: 35 catalog entries fit the 8,000-token budget; 41 sections were omitted and recorded. Only 12 entries were selected. Candidate coverage is still incomplete; navigation is not an exhaustive document review. Nested numbered list items are also detected as boundaries and do not yet have a fully reconstructed legal hierarchy.

A downstream-only run froze both understanding and retrieved evidence, then ran coverage, extraction, answer and verification. It produced a partial answer in 46.562 seconds, cost $0.07512690 including external evaluation. This timing excludes retrieval and cannot establish end-to-end P95.

The external evaluation was rejected: two requirements were marked retrieved=true with empty evidence IDs. No overall accuracy or valid requirement score is reported. Its positive judgments are not professional approval. The answer still omits notice details and exceptions. Runtime verification also removed claims whose subject or procedure did not match the question.

Release remains blocked. Next work is to improve bounded navigation coverage and section hierarchy, and validate each required source passage against final context before generating another broad score. The task is not complete; no user approval of these generated answers has occurred.
