# Locked rule-switch cohort on our own engine (15 seeds)

Locked , before any cohort run.

The 3-seed pilot (`docs/engine-replication-pilot-protocol.md`) replicated by its locked rule: on `js/engine.js` the taught Fly B led the best control at 48 rounds by 44, 23, and 22 wins, and the teacher passed its gate on every seed (64, 60, 65 of 72). This cohort estimates the size of that lead on 15 new seeds. The pilot seeds were seen before this protocol was written, so they are excluded and stay reported as a pilot.

**Hypothesis.** On `js/engine.js`, 48 rounds after the rule change, a Fly B taught by the trained teacher wins more frozen games than every control, on average.

## Design

Identical to the pilot, which is the rule-switch cohort's design (`docs/rule-switch-cohort-protocol.md`) with the networks built from `js/engine.js`: five arms (`trained`, `random`, `untrained`, `shuffled`, `alone`), teacher gate ≥58/72, 96 rounds after the switch, frozen teacher-disconnected checks at 0, 24, 48, and 96. Learner settings and the feature scaling constants are unchanged and not re-fitted. Runner: `tools/check_engine_replication.mjs`, unchanged except that seed indices 3-17 are accepted and written to `data/engine_replication_cohort/`.

## Seeds

The pilot's formulas, for `s` in 3-17: teacher `31001 + 2s`, student `31002 + 2s`; opponent seeds `810000 + 100s + e` (teacher training), `830000 + 100s + e` (old-rule history), `850000 + 100s + e` (switch), `870000 + 10s + e` (teacher gate and frozen checks); shuffle rng `2026111140 + 2s`. None overlap the pilot or earlier runs.

## Fixed analysis

`tools/analyze_engine_replication_cohort.mjs`, written before any cohort run. Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` in frozen wins out of 72 at 48 rounds. Mean advantage, with a 95% percentile bootstrap (10,000 resamples, rng `2026111150`). This is the test the original cohort used.

**Pass:** ≥10 seeds pass the teacher gate, and the bootstrap lower bound is > 4.

**Reported, not tested:** the share of seeds with `advantage <= 4`; mean `trained - alone` at 48 rounds (rng `2026111151`); the mean advantage at 24 and 96 rounds (rng `2026111152`, `2026111153`); mean wins per arm at every checkpoint; seeds that stop at the gate; and, for comparison only, the original cohort's figures on `js/sim.js` (+24.9, 95% CI +17.8 to +32.1).

No seed is replaced, and nothing changes after results are seen.

## Limits

As in the pilot: an engine written within this project by an author who had read `js/sim.js`, so a pass shows the result does not depend on one codebase's numerical choices; it is not an independent replication. The learners still build on Fly Hero's learner classes and the connectome subset is Fly Hero's selection.
