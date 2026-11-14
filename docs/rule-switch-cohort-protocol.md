# Locked rule-switch cohort (15 seeds)

Locked , before any run. The rule-switch confirmation passed 2 of 3 seeds, but on one seed random messages matched the teacher. This cohort estimates, across 15 new seeds, how much a trained teacher beats every control 48 rounds after a rule change, and how often random messages do as well. The design is the confirmation's (`docs/rule-switch-confirmation-protocol.md`) with only seeds and window length changed. Runner: `tools/check_rule_switch_cohort.mjs`, generated from the confirmation runner with those constants changed.

## Seeds

For `s` in 0-14: teacher `26001 + 2s`, student `26002 + 2s`. Opponent seeds: teacher training `110000 + 100s + e`, teacher gate and frozen checks `170000 + 10s + e`, Strategy A history `130000 + 100s + e`, switch `150000 + 100s + e`. Shuffle rng `2026110906 + 2s`. These ranges do not overlap earlier runs or each other.

## Design

Same five arms: trained, random, untrained, shuffled, and alone. 96 switch rounds, with frozen checks at 0, 24, 48, and 96. The teacher gate is ≥58/72.

## Fixed analysis

Per completed seed, `advantage = trained - max(alone, random, untrained, shuffled)` at 48 rounds. Mean advantage, with a 95% percentile bootstrap (10,000 resamples, rng `2026110905`).

**Pass:** ≥10 seeds pass the teacher gate, and the bootstrap lower bound is > 4.

**Reported, not tested:**
- The share of seeds where `trained - random <= 4` at 48 rounds (random matches the teacher).
- The mean `trained - random` at 48 rounds.
- Wins at 96 rounds.
- Seeds that stop at the gate.

No seed is replaced, and nothing changes after results are seen.

## Limits

Fixed RPS rules and one rule change. The untrained teacher may send a constant move, which makes it a weak control.
