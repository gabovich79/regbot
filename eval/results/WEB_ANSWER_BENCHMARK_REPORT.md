# Initial Web Benchmark Report

**Run:** 2026-08-22
**Cases:** 36
**Target:** `https://regbot-wly9.onrender.com/api/chat`
**Raw results:** `eval/results/web_answer_results.jsonl`

## Deterministic scores

| Metric | Result |
|---|---:|
| Strict pass | 1/36 (2.8%) |
| Mean conclusion score | 0.181 |
| Mean required-concepts score | 0.190 |
| Mean actionability score | 0.111 |
| Mean clarification score | 0.764 |
| Answers explicitly reporting insufficient corpus evidence | 7/36 |

These are diagnostic scores, not a claim that the model has 2.8% general accuracy. The first rubric is intentionally strict and lexical; it is designed to expose missing conditions and unsupported claims. A second semantic/manual grading pass is required before using this as a product KPI.

## Per-category results

| Category | Cases | Strict pass | Conclusion | Concepts | Actionability | Clarification |
|---|---:|---:|---:|---:|---:|---:|
| investments | 9 | 0 | 0.278 | 0.167 | 0.333 | 1.000 |
| abstention | 10 | 0 | 0.000 | 0.150 | 0.000 | 0.150 |
| withdrawal | 4 | 0 | 0.250 | 0.125 | 0.000 | 1.000 |
| severance | 3 | 0 | 0.333 | 0.278 | 0.000 | 1.000 |
| employment | 2 | 0 | 0.000 | 0.250 | 0.000 | 1.000 |
| products | 2 | 0 | 0.250 | 0.000 | 0.000 | 1.000 |
| fees | 1 | 0 | 0.000 | 0.000 | 0.000 | 1.000 |
| identity | 1 | 0 | 0.000 | 0.500 | 0.000 | 1.000 |
| reporting | 1 | 0 | 0.000 | 0.500 | 0.000 | 1.000 |
| mobility | 1 | 0 | 0.000 | 0.500 | 0.000 | 1.000 |
| retirement | 1 | 0 | 0.000 | 0.000 | 0.000 | 1.000 |
| operations | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

## Initial findings

### 1. Corpus coverage gaps

The bot explicitly reported insufficient evidence for several questions whose public reference answers are available, including:

- why pension savings are invested in the capital market;
- tax treatment of transfers between investment tracks;
- withdrawal from an investment provident fund;
- management-fee ceilings for an investment provident fund;
- returning from a severance/annuity sequence.

This is mainly a corpus-gap signal: the current corpus contains regulations around these domains but not every explanatory or tax source needed for a complete answer.

### 2. Retrieval drift

Some answers retrieved related documents but missed the question's actual issue. Examples:

- a question about changing an investment track returned fees and transfer mechanics;
- a question about preserving insurance coverage after stopped deposits returned management-fee rules;
- a question about changing tracks was answered with cash-conversion mechanics from a different product.

This is a retrieval/query-understanding problem, not primarily a UI problem.

### 3. Overconfident or weakly grounded framing

The bot sometimes produced a generic conclusion with `CONFIDENCE: HIGH` while citing documents that did not directly answer the public reference question. The answer contract must downgrade confidence when evidence is adjacent rather than direct.

### 4. Abstention quality

The bot often recognized missing information, which is preferable to hallucination. However, the clarification cases show that it does not consistently ask for the missing facts in the exact form required by the case.

## Recommended next engine work

1. Add the missing primary sources for the high-value gaps: investment-provident-fund taxation/liquidity, official tax rules, sequence procedures, and explanatory investment-track material.
2. Add question-understanding fields internally: product, actor, event, requested decision, dates, and constraints.
3. Group evidence by issue before generation instead of passing only top-ranked chunks.
4. Add a groundedness rule: `HIGH` confidence requires direct evidence for the conclusion, not merely related sources.
5. Add a semantic/manual judge pass for the 36 raw answers, then use calibrated thresholds rather than the first strict lexical score as the KPI.
6. Re-run this exact dataset after each engine change.
