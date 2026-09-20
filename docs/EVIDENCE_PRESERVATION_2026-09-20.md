# Evidence preservation diagnostic — 2026-09-20

Code: 204be78. Local copied staged index, no web, no production changes. 243 tests passed.

Removed three prefix/cap losses: binding now validates the entire bounded extracted collection; omitted candidates are no longer crowded out by the generator's 30-claim limit; source-discovered aspects are retained alongside the question plan. Existing 100-unit and 180-component ceilings still reject oversized contracts. All candidates still require source resolution and semantic verification. No question-specific rules were introduced.

Frozen-data replay retained all 23 previously extracted units for fees and annual cost, versus 20 before. The joining fixture retained its original 19. This proves structural preservation only, not semantic correctness.

Two new end-to-end development diagnostics:

| Case | Answer time | Extracted / retained units | Requirements retrieved | Requirements answered (advisory judge) | Previous answered |
|---|---:|---:|---:|---:|---:|
| Fees | 56.2 s | 32 / 32 | 7 / 7 | 6 / 7 | 5 / 7 |
| Joining | 68.8 s | 36 / 36 | 11 / 11 | 2 / 11 | 4 / 11 |

Both answers were partial. Both evaluator responses were structurally valid. The evaluator flagged no incorrect/unsupported claims, but this does not establish professional correctness. Joining regressed and exceeded the 60-second target. Two development cases are not an acceptance or P95 measurement.

Fees now includes transitional provisions but still omits scope exclusions. Joining extraction overemphasizes form details and misses substantive rules, notwithstanding source retrieval and the strengthened general extraction prompt. Preserving an extracted collection does not recover rules absent from that collection. The semantic verifier did not catch all these omissions. Prompt wording alone has not demonstrated a reliable solution.

Release remains blocked. Next implementation should make extraction coverage inspectable by source section and obligation category (operative rule, population, conditions, exceptions, transition, procedure), with source-bound links between limitations and affected rules. Validate this against frozen evidence before further full runs. Do not use evaluation reference requirements as runtime input. Do not claim completeness from approval of the claims that happen to have been extracted.

Latest calls cost $0.18042017. Cumulative committed development budget is $5.27148762 of $20, including $0.58204941 in unsettled reservations; the latter is not confirmed billing. Raw private answers and traces remain local.
