# Web-Derived Professional Answer Benchmark

## Purpose

This benchmark tests whether RegBot answers real professional/regulatory questions, not only whether it retrieves the expected document.

The bot receives **only the question**. It does not receive the web answer or reference URL.

## Dataset composition

- 36 cases total.
- 26 cases seeded from publicly available Q&A / rights / institutional pages.
- 10 clarification/abstention cases derived from the same public topics.
- `official_qa` cases use an official government Q&A page as the reference answer.
- `rights_site` and `institutional_qa` cases are question seeds and must be checked against primary regulation before being treated as legal truth.
- `constructed_from_web_topics` cases test whether the bot asks for missing facts instead of inventing a professional recommendation.

## Case schema

Each JSONL case contains:

- `question`: sent to RegBot;
- `reference_answer`: concise reference answer from the public source;
- `reference_url`: source page;
- `source_type`: provenance class;
- `expected_conclusion`: terms/phrases representing the expected conclusion;
- `required_concepts`: conditions, exceptions, or distinctions that should appear;
- `required_actions`: operational next steps;
- `must_not_include`: dangerous overclaims;
- `requires_clarification` and `clarification_terms` where applicable.

## Scoring

The deterministic scorer reports:

- conclusion score;
- required-concepts score;
- actionability score;
- clarification score;
- citation presence where configured;
- prohibited claims;
- overall strict pass/fail.

Strict pass/fail is intentionally conservative. Partial scores are more useful for diagnosing whether a failure is:

1. wrong conclusion;
2. incomplete answer;
3. not actionable;
4. missing clarification;
5. missing corpus evidence;
6. retrieval drift.

## Run

```bash
.venv/bin/python scripts/run_web_answer_benchmark.py \
  --cases eval/web_answer_cases.jsonl \
  --output eval/results/web_answer_results.jsonl \
  --url https://regbot-wly9.onrender.com/api/chat
```

The command sends only questions to the live API and stores raw answers plus scores. It does not modify the corpus or conversation database.

## Important interpretation rule

A web answer is not automatically the legal truth. For production decisions, validate the reference answer against the primary law/regulation and record the primary source separately. A case that the bot correctly refuses because the source is absent is a **corpus coverage finding**, not automatically a model failure.
