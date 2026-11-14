# Stale-teacher control (3-seed pilot)

Locked , before any run.

## Question

In the rule-switch cohort, the untrained teacher sent one constant move, so it is a weak control. It cannot tell apart two explanations of the taught advantage: (a) Fly B benefits from a teacher that knows the *new* rule, or (b) Fly B benefits from any competent, confident, cue-dependent teacher. A teacher trained to the gate on the *old* rule (Strategy A) separates them: it is just as trained and just as confident, but its knowledge is out of date.

**Hypothesis.** 48 rounds after the rule change, a Fly B taught by the new-rule teacher wins more frozen games than a Fly B taught by the old-rule ("stale") teacher.

## Design

The design is the rule-switch cohort's (`docs/rule-switch-cohort-protocol.md`) with one variable changed: the control arm. Arms: `trained` (teacher trained on Strategy B, as before), `stale` (same network seed and procedure, trained on Strategy A), and `alone`. The random, shuffled, and untrained arms were measured in the 15-seed cohort and are not repeated. Connectome, dynamics, dopamine rule, student, message channel, 96 switch rounds, and frozen checks at 0, 24, 48, and 96 are unchanged. Runner: `tools/check_stale_teacher_control.mjs`, derived from `tools/check_rule_switch_cohort.mjs`.

Both teachers must pass a ≥58/72 frozen gate on the rule they were trained on (B for trained, A for stale). A seed that fails either gate stops and is reported, not replaced.

## Seeds

New seeds, not used by any earlier run. For `s` in 0-2: teacher `27001 + 2s`, student `27002 + 2s`. Opponent seeds: trained-teacher training `210000 + 100s + e`, stale-teacher training `230000 + 100s + e`, Strategy A history `250000 + 100s + e`, trained-teacher gate and frozen checks `270000 + 10s + e`, stale-teacher gate `280000 + 10s + e`, switch `290000 + 100s + e`.

## Fixed analysis

Per completed seed: `trained - stale` and `stale - alone` at 48 rounds, plus all checkpoints. Also reported: how often the stale teacher's message equals the trained teacher's on the same round.

**Supports the hypothesis:** `trained - stale > 4` at 48 rounds on all completed seeds, with at least 2 completed.
**Does not support it:** anything else.

Three seeds is the project's exploratory limit. This is a pilot: no confidence interval, and it is reported as a pilot whatever the outcome. Nothing changes after results are seen.

## Limits

Fixed RPS rules and one rule change. The stale teacher's advice matches what the student already learned, so it may slow the student below `alone`; that would show stale advice is harmful here, not merely unhelpful.
