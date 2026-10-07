# Annex boundary audit — 2026-09-20

Correction to earlier diagnosis: the automated judge's statement that all required joining evidence was retrieved is not reliable. Inspection of the actual selected source pages showed predominance of form appendices. Do not interpret the previous 11/11 retrieval score as established evidence coverage. Exact ID validation prevents fabricated pointers, but does not prove semantic entailment.

Confirmed implementation defect: `knowledge.SECTION` did not recognize annex headings, including extracted Hebrew headings with a leading apostrophe. This made 39 chunks inherit section `19. ביטול חוזרים`. Parent expansion then treated that long run of annex material as the same legal section.

Fixed heading recognition and annex parent paths. Five regression cases check heading variants, nested numbered headings, inline references, and exact preservation of short source text. Full suite: 248 passed.

Offline rebuild of the original joining text: source hash unchanged, 70 to 76 chunks, cancellation section reduced from 39 chunks to 1 (427 characters). A new version was staged on a separate SQLite copy, retaining old versions. Version `f452819ea4bd44bca490f13e9ab55ba0`, 76 chunks. No index activation or production modification. Staging cost $0.02219133.

Controlled retrieval-only run used the saved question plan and newly staged document plus the other existing diagnostic versions. Cost $0.00690289. It returned 20 source chunks with correct annex labels, but still predominantly annex material; it did not demonstrate recovery of all required operative provisions. No new answer-quality or acceptance score is claimed. No fresh full answer run was justified by this retrieval result.

Remaining blocker: document/section discovery must bring operative provisions and applicable scope into candidates alongside form fields; the verifier also needs trustworthy source-support evaluation. Correcting headings is necessary but insufficient. Next work should inspect candidate coverage before extraction or generation, using the corrected document structure and original source evidence, without test-question-specific ranking rules.

Earlier reports remain as historical records, not release evidence. Release is still blocked. The original database has not been deleted or replaced.
