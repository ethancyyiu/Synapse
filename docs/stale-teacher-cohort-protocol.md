# Locked stale-teacher cohort (15 seeds)

Locked , before any cohort run. The 3-seed pilot (`docs/stale-teacher-control-protocol.md`) supported its hypothesis on 3 of 3 seeds: a teacher trained on the old rule gave the student no benefit. This cohort estimates the size of that difference on 15 new seeds. The pilot seeds were seen before this protocol was written, so they are excluded from the cohort analysis and stay reported as a pilot.

## Design

Identical to the pilot: arms `trained` (teacher trained on Strategy B), `stale` (same network seed and procedure, trained on Strategy A), and `alone`; both teachers gated at ≥58/72 on their own rule; 96 switch rounds with frozen checks at 0, 24, 48, and 96. Runner: `tools/check_stale_teacher_control.mjs`, unchanged except that seed indices 3-17 are accepted and written to `data/stale_teacher_cohort/`.

## Seeds

The pilot's formulas, for `s` in 3-17: teacher `27001 + 2s`, student `27002 + 2s`; opponent seeds `210000 + 100s + e` (trained-teacher training), `230000 + 100s + e` (stale-teacher training), `250000 + 100s + e` (Strategy A history), `270000 + 10s + e` (trained gate and frozen checks), `280000 + 10s + e` (stale gate), `290000 + 100s + e` (switch). None overlap the pilot or earlier runs.

## Fixed analysis

`tools/analyze_stale_teacher_cohort.mjs`, written before any cohort run. Per completed seed, `difference = trained - stale` at 48 rounds. Mean difference, with a 95% percentile bootstrap (10,000 resamples, rng `2026111001`).

**Pass:** ≥10 seeds complete (both teachers pass their gates), and the bootstrap lower bound is > 4.

**Reported, not tested:**
- Mean `stale - alone` at 48 and 96 rounds, with the same bootstrap (rng `2026111002`), and the share of seeds where stale is more than 4 wins below alone at 96 rounds.
- Mean wins per arm at every checkpoint.
- How often the two teachers' messages agree.
- Seeds that stop at a gate.

No seed is replaced, and nothing changes after results are seen.

## Limits

Fixed RPS rules and one rule change. The old-rule teacher's advice matches what the student already learned, so this tests out-of-date knowledge specifically, not every possible wrong teacher.
