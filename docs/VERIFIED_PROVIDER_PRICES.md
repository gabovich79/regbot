# Provider prices verified 2026-09-19

`deployment/provider-prices-2026-09-19.json` is an explicitly selected configuration snapshot, not an automatically enabled default. Input/output prices are USD per million tokens. Reverify before public rollout or any model change.

- Gemini 2.5 Flash standard paid text: $0.30 input and $2.50 output, including thinking. Google Search grounding: $35 per 1,000 grounded prompts above the free allowance. Reserve $0.035 per discovery call regardless of free allowance; this is conservative cost accounting. [Google pricing](https://ai.google.dev/gemini-api/docs/pricing).
- Gemini 2.5 grounding is billed per prompt; the per-search-query billing rule applies to Gemini 3. Do not carry the per-call ceiling to another model without reviewing its billing. [Google Search billing](https://ai.google.dev/gemini-api/docs/google-search).
- text-embedding-3-large standard embedding input: $0.13 per million tokens. [OpenAI model pricing](https://developers.openai.com/api/docs/models/text-embedding-3-large).

This verifies published rates, not account-specific access, credits or provider-level spending controls. Public use remains subject to the application reservation ledger and deployment checks. Indexing and evaluation use their separately recorded purpose and explicit batch caps.
