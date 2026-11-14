# Rule-switch teaching on our own engine (3-seed pilot)

Locked , before any run.

## Question

Every result so far was produced with `js/sim.js`, the neural simulator that came with Fly Hero. `js/engine.js` is a simulator written for this project: the same published neuron model (Shiu et al. 2024) and the same connectome file, but integrated exactly instead of by 1 ms Euler steps, and with random numbers consumed in a different order. Its spike trains are not the same as `js/sim.js`'s. Does the headline teaching result still appear on it?

What was seen before locking, on development trials that used no seed below (`node tools/test_engine.mjs --compare`): both engines read the cue from the sensory lanes on 36 of 36 trials and the advice code from the two message neurons on 36 of 36. The output-neuron (MBON) rates the learners read are lower on the new engine, for example 0.12 against 0.52 spikes per neuron for cue 0, because exact integration gives a smaller response to each input than the Euler step does. So a pass is not assured: the teacher, which reads only those three rates, may not learn well enough to pass its gate.

**Hypothesis.** On `js/engine.js`, 48 rounds after the rule change, a Fly B taught by the trained teacher wins more frozen games than every control.

## The one change

The engine. The design is the rule-switch cohort's (`docs/rule-switch-cohort-protocol.md`): the same Rock-Paper-Scissors rules, five arms (`trained`, `random`, `untrained`, `shuffled`, `alone`), teacher gate ≥58/72, 4 history episodes, 96 rounds after the switch, frozen teacher-disconnected checks at 0, 24, 48, and 96. The learners, their settings, the message code, the schedule, and the feature scaling constants (`featureMean`, `featureStd`, measured on `js/sim.js`) are all unchanged and are not re-fitted for the new engine.

Runner: `tools/check_engine_replication.mjs`, derived from `tools/check_rule_switch_cohort.mjs` with the network built from `js/engine.js`. `frozenStudentCheckpoint` gains an optional `engine` argument that defaults to `js/sim.js`, so every existing caller behaves as before.

## Seeds

New, unused by any earlier run. For `s` in 0-2: teacher `31001 + 2s`, student `31002 + 2s`. Opponent seeds: teacher training `810000 + 100s + e`, old-rule history `830000 + 100s + e`, switch `850000 + 100s + e`, teacher gate and frozen checks `870000 + 10s + e`. Shuffle rng `2026111140 + 2s`.

## Fixed analysis

Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` in frozen wins out of 72 at 48 rounds.

**Replicates:** `advantage > 4` on every completed seed, with at least 2 completed.
**Does not replicate as it stands:** anything else, including fewer than 2 seeds passing the teacher gate.

Reported, not tested: every checkpoint for every arm, and the teacher's frozen score.

Three seeds, a pilot with no confidence interval, reported as a pilot whatever the outcome. Nothing changes after results are seen. If the teacher fails its gate, the scaling constants are not re-fitted within this pilot; that would be a separate, later test.

## Limits

One engine written by the same project, by an author who had read `js/sim.js`, so this is a check that the result does not depend on one codebase's numerical choices, not an independent replication. The learners still build on Fly Hero's learner classes, and the connectome subset is Fly Hero's selection from the Janelia data.
