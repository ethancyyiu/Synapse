# Locked fault-response cohort, full change (15 seeds)

Locked , before any cohort run.

## Where this comes from, and what to hold against it

Two 3-seed pilots on this task **did not support** carry-over. With a small change, Fly B alone was perfect within 24 rounds and the teacher added nothing (`docs/fault-response-pilot-protocol.md`). With a full change, the taught Fly B led the best control at 48 rounds by 25, 0, and 25 wins, and the rule required a lead on every seed (`docs/fault-response-full-change-protocol.md`). That second protocol said the line would stop if it failed. The project owner asked for it to be reopened as a full-size test, so this cohort exists by that decision, and the earlier stop is recorded here so a reader can weigh it.

Three choices below were made after seeing the pilots: running a cohort at all, keeping the full-change table, and testing the mean advantage with an interval in place of an every-seed rule. The mean-based test is the one the Rock-Paper-Scissors rule-switch cohort used (`docs/rule-switch-cohort-protocol.md`), where a 2-of-3 confirmation was likewise followed by a 15-seed cohort. No pilot seed is reused, and both pilots stay reported as failed.

**Hypothesis.** On the fault-response task with a full change, 48 rounds after the change a Fly B taught by the trained teacher wins more frozen rounds than every control, on average.

## Design

Identical to the full-change pilot: alarms in random order; fixing responses 1, 1, 2 before the change and 2, 2, 0 after it; five arms (`trained`, `random`, `untrained`, `shuffled`, `alone`); teacher gate ≥58/72; 4 history episodes; 96 rounds after the change; frozen teacher-disconnected checks at 0, 24, 48, and 96. Runner: `tools/check_fault_response_pilot.mjs`, unchanged except that seed indices 20-34 are accepted and written to `data/fault_response_cohort/`.

## Seeds

New. The pilots' formulas with `s` in 20-34: teacher `30001 + 2s`, student `30002 + 2s`; alarm-order seeds `710000 + 100s + e` (teacher training), `730000 + 100s + e` (history), `750000 + 100s + e` (after the change), `770000 + 10s + e` (teacher gate and frozen checks); shuffle rng `2026111120 + 2s`. None overlap the pilots (indices 0-2 and 10-12) or earlier runs.

## Fixed analysis

`tools/analyze_fault_response_cohort.mjs`, written before any cohort run. Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` in frozen wins out of 72 at 48 rounds. Mean advantage, with a 95% percentile bootstrap (10,000 resamples, rng `2026111130`).

**Pass:** ≥10 seeds pass the teacher gate, and the bootstrap lower bound is > 4.

**Reported, not tested:** the share of seeds with `advantage <= 4`; mean `trained - alone` at 48 rounds with the same bootstrap (rng `2026111131`); the mean advantage at 24 and 96 rounds (rng `2026111132`, `2026111133`); mean wins per arm at every checkpoint; seeds that stop at the gate.

No seed is replaced, and nothing changes after results are seen.

## Limits

One synthetic task with fixed outcomes: three cues, three choices, on the same simulated connectome, learners, and message channel as every other study here. It is not a machine, a robot, or real sensor data. The table was picked to resemble a change known to be hard to recover from alone, so a pass would show the teaching effect is not specific to the Rock-Paper-Scissors cycle; it would not show that teaching helps after changes in general, and the small-change pilot shows it adds nothing when recovery alone is quick. This is the third test on this task.
