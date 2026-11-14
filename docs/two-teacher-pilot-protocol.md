# Two-teacher pilot (3 seeds)

Locked , before any run.

## Question

So far Fly B has had one teacher. In practice an adapting system may hear from several partners, some of them out of date, without being told which is which. If Fly B hears a new-rule teacher and an old-rule teacher on every round, can it work out from its own wins and losses which one to follow?

**Hypothesis.** A Fly B that keeps a separate trust value for each teacher (the rule from `docs/trust-gate-pilot-protocol.md`, unchanged) recovers after the rule change faster than a Fly B that follows both teachers equally, and about as fast as a Fly B that hears only the new-rule teacher.

## The one change

The number of teachers: two messages per round in place of one. Everything else is as in the trust-gate studies.

- Both teachers send their preferred move as spikes on the same two descending neurons (DNge175), one after the other: slot 0 at 240-280 ms, slot 1 at 290-330 ms. Each slot is decoded with the existing count code. The runner stops with an error if a decoded message differs from the one sent.
- Which teacher uses which slot is fixed within a run and set by the seed: the new-rule teacher has slot 0 on even seed indices and slot 1 on odd ones. Fly B is never told which slot holds which teacher.
- Fly B plays its own move before either message, learns from its own reward, then applies one imitation step per message.
- `bothGated` keeps one trust value per slot, each starting at 1 and updated by the locked rule: on a round where Fly B had already played that slot's advised move, `trust += 0.2 * (won - trust)`; the slot's imitation step is `0.03 * trust`. The rate 0.2 is the one fixed in the trust-gate protocol and is not tuned.
- `bothUngated` applies the full `0.03` step for each message.

The two-teacher round and student live in the runner, `tools/check_two_teacher_pilot.mjs` (derived from `tools/check_trust_gate_pilot.mjs`). `PreferenceStudent`, the demo, the connectome, the dynamics, and the dopamine rule are unchanged.

## Design

Arms, all starting from the same old-rule history: `alone`; `trained` (the new-rule teacher only, ungated, as in the earlier cohorts); `bothUngated`; `bothGated`. Both teachers must pass the ≥58/72 gate on their own rule; a seed that fails stops and is reported, not replaced. 96 switch rounds, frozen teacher-disconnected checks at 0, 24, 48, and 96. The new-rule teacher's messages in the two-teacher arms must equal those in the `trained` arm on every round, or the runner stops.

## Seeds

New, unused by any earlier run. For `s` in 0-2: teacher `29001 + 2s`, student `29002 + 2s`. Opponent seeds: trained-teacher training `610000 + 100s + e`, stale-teacher training `630000 + 100s + e`, old-rule history `650000 + 100s + e`, trained gate and frozen checks `670000 + 10s + e`, stale gate `680000 + 10s + e`, switch `690000 + 100s + e`.

## Fixed analysis

Per completed seed, frozen wins out of 72 at 48 rounds:

1. **Choosing helps.** `bothGated - bothUngated > 4`.
2. **A bad adviser costs little.** `bothGated - trained >= -4`.

**Supports the hypothesis:** both hold on every completed seed, with at least 2 completed.
**Does not support it:** anything else.

Reported, not tested: every checkpoint for every arm, `bothGated - alone`, and both trust values at each checkpoint.

Three seeds is the project's exploratory limit. This is a pilot with no confidence interval, reported as a pilot whatever the outcome. Nothing changes after results are seen.

## Limits

Fixed RPS rules, one rule change, two teachers, and one kind of wrong teacher whose advice matches what Fly B already plays, so its trust is tested on almost every round. The two-teacher arms take two imitation steps per round where `trained` takes one. Choosing among advisers by their payoff is a known idea; this tests it in this system.
