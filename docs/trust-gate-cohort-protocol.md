# Locked trust-gate cohort (15 seeds)

Locked , before any cohort run. The 3-seed pilot (`docs/trust-gate-pilot-protocol.md`) supported its hypothesis on 3 of 3 seeds: a Fly B that scales imitation by a trust value learned from its own rewards was not held back by the old-rule teacher and kept the benefit of the new-rule teacher. This cohort estimates those differences on 15 new seeds. The pilot seeds were seen before this protocol was written, so they are excluded from the cohort analysis and stay reported as a pilot.

## Design

Identical to the pilot: arms `alone`, `trained`, `stale`, `gatedTrained`, `gatedStale`; the trust rule and its rate 0.2 unchanged; both teachers gated at ≥58/72 on their own rule; 96 switch rounds with frozen teacher-disconnected checks at 0, 24, 48, and 96. Runner: `tools/check_trust_gate_pilot.mjs`, unchanged except that seed indices 3-17 are accepted and written to `data/trust_gate_cohort/`.

## Seeds

The pilot's formulas, for `s` in 3-17: teacher `28001 + 2s`, student `28002 + 2s`; opponent seeds `410000 + 100s + e` (trained-teacher training), `430000 + 100s + e` (stale-teacher training), `450000 + 100s + e` (old-rule history), `470000 + 10s + e` (trained gate and frozen checks), `480000 + 10s + e` (stale gate), `490000 + 100s + e` (switch). None overlap the pilot or earlier runs.

## Fixed analysis

`tools/analyze_trust_gate_cohort.mjs`, written before any cohort run. Over completed seeds, three mean differences in frozen wins out of 72, each with a 95% percentile bootstrap (10,000 resamples):

1. **Protection.** `gatedStale - stale` at 96 rounds (rng `2026111101`). Holds if the lower bound is > 4.
2. **No harm.** `gatedStale - alone` at 96 rounds (rng `2026111102`). Holds if the lower bound is > -4.
3. **Benefit kept.** `gatedTrained - trained` at 48 rounds (rng `2026111103`). Holds if the lower bound is > -4.

**Pass:** ≥10 seeds complete (both teachers pass their gates), and all three hold.

**Reported, not tested:** mean wins per arm at every checkpoint; `gatedTrained - alone` at 48 rounds with the same bootstrap (rng `2026111104`); mean trust of each gated student at each checkpoint; seeds that stop at a gate.

No seed is replaced, and nothing changes after results are seen.

## Limits

As in the pilot: fixed RPS rules, one rule change, and one kind of wrong teacher whose advice Fly B tests on almost every round. Weighing advice by its payoff is a known idea in action advising.
