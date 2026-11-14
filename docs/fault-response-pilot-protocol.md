# Fault-response task pilot (3 seeds)

Locked , before any run.

## Question

Every teaching result so far is on Rock-Paper-Scissors against an opponent that cycles through its moves. Does the same teaching help on a second task with a different structure, closer to the uses listed on the Research tab?

Three earlier second-task attempts stopped before any teaching was tried: continuous sensor readings did not survive the sensory pathway (`docs/sensor-classification-report.md`), a multi-step route was not learned by the teacher (`docs/warehouse-navigation-report.md`), and neither was a task with random outcomes (`docs/package-recovery-report.md`). This task stays inside what those studies showed the system can do: three distinct cues, one decision per round, a fixed outcome for each decision.

## The task

A machine raises one of three alarms. The fly picks one of three responses. For each alarm one response fixes the fault (+1), one does nothing (0), and one makes it worse (-1).

| | Alarm 0 | Alarm 1 | Alarm 2 |
|---|---|---|---|
| Fixing response before the change | 1 | 1 | 2 |
| Fixing response after the change | 1 | **0** | 2 |

What differs from the Rock-Paper-Scissors studies:

- **Alarms arrive in random order.** Each round's alarm is drawn independently and uniformly. In Rock-Paper-Scissors the cue followed a fixed cycle set by the opponent.
- **Two alarms share a response before the change**, and one response is never right. The Rock-Paper-Scissors rules always paired each cue with a different move.
- **The change is partial.** Only alarm 1's fixing response changes. Two thirds of what Fly B knows stays correct, so a frozen test starts near 48 of 72, not near 0, and a message can now do damage by disturbing what was right.
- There is a cue on every round, so there is a message on every round.

What does not differ, stated plainly: it is still three cues and three choices with a fixed win/nothing/worse outcome, run on the same simulated connectome, sensory lanes, learners, message channel, and schedule. It is a toy decision task with alarm labels, not a machine, a robot, or real sensor data.

Implementation: `AlarmGame` in the runner stands in for the opponent object. The alarm is presented on the sensory lane the earlier studies used for that cue index, and the outcome uses the existing `result()` function with the alarm's hidden "needed" value in place of the opponent's move. `js/` is unchanged.

## Design

The rule-switch cohort design (`docs/rule-switch-cohort-protocol.md`) with one variable changed: the task. Five arms from the same before-the-change history: `trained`, `random`, `untrained`, `shuffled`, `alone`. The teacher is trained on the after-the-change table for 12 episodes of 24 rounds and must win ≥58/72 on a frozen test, or the seed stops and is reported, not replaced. Fly B history: 4 episodes of 24 rounds on the before-the-change table. Then 96 rounds after the change, with frozen teacher-disconnected checks of 72 rounds at 0, 24, 48, and 96. Runner: `tools/check_fault_response_pilot.mjs`, derived from `tools/check_rule_switch_cohort.mjs`.

## Seeds

New, unused by any earlier run. For `s` in 0-2: teacher `30001 + 2s`, student `30002 + 2s`. Alarm-order seeds: teacher training `710000 + 100s + e`, Fly B history `730000 + 100s + e`, after the change `750000 + 100s + e`, teacher gate and frozen checks `770000 + 10s + e`. Shuffle rng `2026111120 + 2s`.

## Fixed analysis

Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` in frozen wins out of 72 at 48 rounds.

**Supports carrying over:** `advantage > 4` on every completed seed, with at least 2 completed.
**Does not support it:** anything else, including fewer than 2 seeds passing the teacher gate.

Reported, not tested: every checkpoint for every arm; each arm's frozen wins split into the changed alarm and the two unchanged alarms; the teacher's frozen score.

Three seeds is the project's exploratory limit. This is a pilot with no confidence interval, reported as a pilot whatever the outcome. Nothing changes after results are seen.

## Limits

One synthetic task, fixed outcomes, one partial change. With 48 of 72 frozen wins available before any relearning, the largest possible advantage is about 24, where it was about 72 in Rock-Paper-Scissors. A pass would show the teaching effect is not specific to the Rock-Paper-Scissors cycle; it would not show that any real system benefits.
