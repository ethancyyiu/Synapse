# Trust-gate pilot (3 seeds)

Locked , before any run.

## Question

The stale-teacher cohort (`docs/stale-teacher-cohort-protocol.md`) showed that Fly B follows advice without checking it: a teacher trained on the old rule held its students 30 wins below learning alone at 96 rounds. Can Fly B use its own wins and losses to decide how much to trust its teacher, so that an out-of-date teacher no longer harms it and a current teacher still helps?

**Hypothesis.** A Fly B that scales its imitation step by a trust value learned from its own rewards (a) is not held below learning alone by the old-rule teacher, and (b) keeps the benefit of the new-rule teacher.

## The one change

The stale-teacher design with one variable changed: the student's imitation step is multiplied by `trust`. `TrustGatedStudent` in `js/rps_challenge_students.js` is an opt-in subclass; `PreferenceStudent`, the demo, the connectome, the dynamics, the dopamine rule, the reward update, and the message channel are unchanged.

- `trust` starts at 1 (the existing student's behaviour).
- On a round where a message arrived and the move Fly B had already played equals the advised move: `trust += 0.2 * (won - trust)`, with `won` 1 for a win and 0 for a draw or loss.
- On other rounds `trust` does not change.
- The imitation step is `0.03 * trust` in place of `0.03`.

Fly B uses only its own move, its own reward, and the decoded message. It is never told the opponent's rule, the correct move, or which teacher it has. The rate 0.2 is fixed here and is not tuned.

## Design

Arms, all starting from the same old-rule history: `alone`, `trained` and `stale` (ungated, as in the cohort), `gatedTrained` and `gatedStale` (trust-gated, same teachers). The gated arms receive the same messages on the same rounds as the ungated arms; the runner stops with an error if they differ. Both teachers must pass the ≥58/72 gate on their own rule; a seed that fails stops and is reported, not replaced. 96 switch rounds, frozen teacher-disconnected checks at 0, 24, 48, and 96. Runner: `tools/check_trust_gate_pilot.mjs`, derived from `tools/check_stale_teacher_control.mjs`.

## Seeds

New, unused by any earlier run. For `s` in 0-2: teacher `28001 + 2s`, student `28002 + 2s`. Opponent seeds: trained-teacher training `410000 + 100s + e`, stale-teacher training `430000 + 100s + e`, old-rule history `450000 + 100s + e`, trained gate and frozen checks `470000 + 10s + e`, stale gate `480000 + 10s + e`, switch `490000 + 100s + e`.

## Fixed analysis

Per completed seed, frozen wins out of 72:

1. **No harm.** `gatedStale - alone >= -4` at 48 and at 96 rounds.
2. **Benefit kept.** `gatedTrained - trained >= -4` at 48 rounds.
3. **Protection is real.** On every seed where the ungated stale student is harmed (`alone - stale > 4` at 96 rounds), `gatedStale - stale > 4` at 96 rounds.

**Supports the hypothesis:** all three hold on every completed seed, with at least 2 completed and at least 1 seed where the ungated stale student is harmed.
**Inconclusive:** 1 and 2 hold but no seed has a harmed ungated stale student.
**Does not support it:** anything else.

Reported, not tested: every checkpoint for every arm, and each gated student's `trust` at each checkpoint.

Three seeds is the project's exploratory limit. This is a pilot with no confidence interval, reported as a pilot whatever the outcome. Nothing changes after results are seen.

## Limits

Fixed RPS rules, one rule change, one kind of wrong teacher. The old-rule advice matches what Fly B already plays, so Fly B tests the advice on almost every round; a wrong teacher whose advice Fly B rarely happens to play would be discounted more slowly. Learning to weigh advice by its payoff is a known idea in action advising; this tests it in this system, not a new method.
