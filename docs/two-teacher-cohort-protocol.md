# Locked two-teacher cohort (15 seeds)

Locked , before any cohort run.

## Where this comes from

The 3-seed pilot (`docs/two-teacher-pilot-protocol.md`) **did not support** its hypothesis. Its locked test was at 48 rounds: the trust-keeping Fly B beat the Fly B that follows both teachers equally on 2 of 3 seeds, and stayed within 4 wins of the single-teacher Fly B on 1 of 3. At 96 rounds, which the pilot reported but did not test, the trust-keeping Fly B equalled the single-teacher Fly B on 3 of 3 seeds (71, 69, 70) and was 12, 25, and 24 wins above the equal-follower.

This cohort tests that 96-round pattern on 15 new seeds. **The 96-round endpoint was chosen after seeing the pilot.** That is why the pilot seeds are excluded and why this is a new test, not a rescue of the pilot, which stays reported as failed.

**Hypothesis.** By 96 rounds after the rule change, a Fly B that hears a new-rule and an old-rule teacher and keeps a trust value for each does better than one that follows both equally, and about as well as one that hears only the new-rule teacher.

## Design

Identical to the pilot: arms `alone`, `trained`, `bothUngated`, `bothGated`; two message slots on DNge175; the trust rule and its rate 0.2 unchanged; both teachers gated at ≥58/72 on their own rule; 96 switch rounds with frozen teacher-disconnected checks at 0, 24, 48, and 96. The new-rule teacher has slot 0 on even seed indices and slot 1 on odd ones. Runner: `tools/check_two_teacher_pilot.mjs`, unchanged except that seed indices 3-17 are accepted and written to `data/two_teacher_cohort/`.

## Seeds

The pilot's formulas, for `s` in 3-17: teacher `29001 + 2s`, student `29002 + 2s`; opponent seeds `610000 + 100s + e` (trained-teacher training), `630000 + 100s + e` (stale-teacher training), `650000 + 100s + e` (old-rule history), `670000 + 10s + e` (trained gate and frozen checks), `680000 + 10s + e` (stale gate), `690000 + 100s + e` (switch). None overlap the pilot or earlier runs.

## Fixed analysis

`tools/analyze_two_teacher_cohort.mjs`, written before any cohort run. Over completed seeds, two mean differences in frozen wins out of 72 at 96 rounds, each with a 95% percentile bootstrap (10,000 resamples):

1. **Choosing helps.** `bothGated - bothUngated` (rng `2026111111`). Holds if the lower bound is > 4.
2. **A bad adviser costs little.** `bothGated - trained` (rng `2026111112`). Holds if the lower bound is > -4.

**Pass:** ≥10 seeds complete (both teachers pass their gates), and both hold.

**Reported, not tested:** the same two differences at 48 rounds (the pilot's failed endpoint) and at 24 rounds, with the same bootstrap (rng `2026111113` to `2026111116`); `bothGated - alone` at 96 rounds (rng `2026111117`); mean wins per arm and mean trust in each teacher at every checkpoint; the two contrasts split by which slot held the new-rule teacher; seeds that stop at a gate.

No seed is replaced, and nothing changes after results are seen.

## Limits

As in the pilot: fixed RPS rules, one rule change, two teachers, and one kind of wrong teacher whose advice Fly B tests on almost every round. The two-teacher arms take two imitation steps per round where `trained` takes one. The endpoint was chosen from pilot data. Choosing among advisers by their payoff is a known idea; this tests it in this system.
