# Teacher learning on our own engine: re-measured input scaling (15 seeds)

Locked , before any run.

## Question

In the rule-switch cohort on `js/engine.js` (`docs/engine-replication-cohort-protocol.md`), Fly A missed its 58-of-72 bar on 5 of 15 seeds; on `js/sim.js` it missed on 0 of 15. Fly A's learner reads three output-neuron (MBON) rates after shifting and scaling each one by two constants, `featureMean` and `featureStd`. Those constants were measured on `js/sim.js`, where the rates are several times higher. Does re-measuring them on `js/engine.js` make Fly A learn as reliably there?

**Hypothesis.** With the two constants re-measured on `js/engine.js`, Fly A passes its bar on more seeds than with the `js/sim.js` constants, and on at least 13 of 15.

## The one change

The two scaling constants. They are re-measured without any reward, outcome, or label, by a fixed procedure:

- A network on `js/engine.js` with seed 32999 and an untrained, non-learning copy of Fly A's learner plays 6 episodes of 24 rounds against the new-rule opponent (opponent seeds `890000 + e`), with the same per-round random streams as teacher training (`key = 32999 * 1e7 + e * 1000 + round`).
- On every round that has a cue, the three MBON group rates the learner read are recorded. `featureMean[j]` is the mean of `log(1 + rate_j)` over those rounds and `featureStd[j]` is its standard deviation (population form). This is the transformation the learner already applies; only the two numbers change.

Everything else is the cohort's teacher procedure: 12 training episodes of 24 rounds on the new rule, then a frozen 72-game test, learning rate and exploration unchanged. No student is trained and no teaching is tested here.

## Design

For each seed, two teachers with the same network seed, opponents, and random streams: `simConstants` (the constants from `data/rps_contextual_bandit.json`, as in the cohort) and `engineConstants` (re-measured as above). Runner: `tools/check_engine_teacher_scaling.mjs`, one write-once file per seed in `data/engine_teacher_scaling/`.

All 15 seeds are run at once, not 3 first: each seed trains two teachers only, which takes about a minute, and a pass rate cannot be read from 3 seeds.

## Seeds

New, unused by any earlier run. For `s` in 0-14: teacher `32001 + 2s`. Opponent seeds: training `910000 + 100s + e`, frozen test `970000 + 10s + e`.

## Fixed analysis

`tools/analyze_engine_teacher_scaling.mjs`, written before any run. Count the seeds on which each teacher scores ≥58 of 72.

**Supported:** `engineConstants` passes on at least 13 of 15 seeds and on more seeds than `simConstants`.
**Not supported:** anything else.

Reported, not tested: the two sets of constants; each teacher's score on every seed; the mean score of each arm; the seeds where the two arms differ.

Nothing changes after results are seen. In particular, the measuring procedure above is not adjusted and no other learner setting is tried.

## Limits

This tests teacher learning only. If it is supported, the teaching studies on `js/engine.js` would still need to be re-run with the new constants before any teaching claim uses them, and the completed engine cohort stays as reported with the old constants. One fixed re-measurement is tried; a different procedure might give different constants.
