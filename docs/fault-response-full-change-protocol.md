# Fault-response task, full change (3-seed pilot)

Locked , before any run of these seeds.

## Where this comes from

The first fault-response pilot (`docs/fault-response-pilot-protocol.md`) **did not support** carry-over: the advantage of the taught Fly B over the best control at 48 rounds was 0 on 3 of 3 seeds. Fly B learning alone reached 72 of 72 by the first checkpoint (24 rounds) on every seed, so a teacher had nothing to add. That pilot differed from the Rock-Paper-Scissors rule switch in several ways at once: random alarm order, a shared response, a change to one alarm only, and an old response that became harmful (-1), where the old Rock-Paper-Scissors move only stopped winning (0).

This pilot keeps the new task and its random alarm order but makes the change like the Rock-Paper-Scissors one in the two remaining respects, so the question is narrower: **with a change as large as the Rock-Paper-Scissors one, does teaching help when the cues come in random order?** The first pilot stays reported as it is.

## The one change from the first pilot

The after-the-change table.

| | Alarm 0 | Alarm 1 | Alarm 2 |
|---|---|---|---|
| Fixing response before the change (as in the first pilot) | 1 | 1 | 2 |
| Fixing response after the change | 2 | 2 | 0 |

Every alarm's fixing response moves to the next response. As in the Rock-Paper-Scissors rule switch, the old response then does nothing (0); it does not make things worse. This table was chosen before any run to mirror the Rock-Paper-Scissors change. It was not tried against alternatives, and it is not adjusted afterwards whatever the outcome.

Everything else is the first pilot's: five arms (`trained`, `random`, `untrained`, `shuffled`, `alone`), teacher gate ≥58/72, 4 history episodes, 96 rounds after the change, frozen teacher-disconnected checks at 0, 24, 48, and 96. Runner: `tools/check_fault_response_pilot.mjs`, unchanged except that seed indices 10-12 use this table and write to `data/fault_response_full_change/`.

## Seeds

New. The first pilot's formulas with `s` in 10-12: teacher `30001 + 2s`, student `30002 + 2s`; alarm-order seeds `710000 + 100s + e`, `730000 + 100s + e`, `750000 + 100s + e`, `770000 + 10s + e`; shuffle rng `2026111120 + 2s`.

## Fixed analysis

Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` in frozen wins out of 72 at 48 rounds.

**Supports carrying over:** `advantage > 4` on every completed seed, with at least 2 completed.
**Does not support it:** anything else.

Reported, not tested: every checkpoint for every arm, and the teacher's frozen score.

Three seeds, a pilot with no confidence interval, reported as a pilot whatever the outcome. If it is supported, the next step is a locked 15-seed cohort with the same table. If it is not, the fault-response line stops: this is its second test.

## Limits

One synthetic task with fixed outcomes. This is the second table tried on this task, and it was picked to resemble a change already known to be hard to recover from alone, so a pass shows the teaching effect survives random cue order and a shared response; it does not show that teaching helps after changes in general. The first pilot shows it adds nothing when recovery alone is quick.
