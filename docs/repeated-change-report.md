# Study 4 result: teaching does not help when the rule changes back, because Fly B recovers an old rule on its own

**Verdict under the locked rule: FAIL.** 48 rounds after the second change (rule B back to rule A), the taught Fly B did not lead the best control (mean -0.07 wins, 95% CI -0.21 to 0.00; needed a lower bound above +4). [Protocol](repeated-change-protocol.md), locked before any run. Simulated Rock-Paper-Scissors only.

## What was run

- Rule-A history, then **change 1** to rule B for 48 rounds, then **change 2** back to rule A for 48 rounds. Own-action reward every round, advice on every observed-cue round as measured DN spikes, frozen teacher-disconnected checks at 0 and 48 (rule B) and 72 and 96 (rule A).
- The current teacher in each phase: one teacher trained on rule B, one on rule A, same network seed, each gated at 58/72.
- Arms: no advice, trained (current teacher), random moves at the same times, shuffled (current teacher's moves permuted within each phase).
- **15 fresh seeds (0-14, new formulas).** 14 completed. **Seed 9 stopped at teacher readiness** (its rule-B teacher scored 57/72, one short of the gate); it was not replaced. All messages decoded in every completed seed. Implementation check seed 20 excluded.

## Results (frozen, teacher disconnected; mean wins out of 72, 14 seeds)

| Checkpoint | Rule | No advice | Trained | Random | Shuffled |
|---|---|---:|---:|---:|---:|
| 0 | B | 1.07 | 1.07 | 1.07 | 1.07 |
| 48 (end of change 1) | B | 16.21 | **69.50** | 34.93 | 33.50 |
| 72 (24 rounds after change 2) | A | **69.43** | 65.57 | 55.93 | 58.86 |
| 96 (48 rounds after change 2) | A | 69.43 | 69.36 | 62.71 | 67.36 |

Trained minus the best of no advice, random and shuffled (95% percentile bootstrap, 10,000 resamples, seed 2026111540):

| Checkpoint | Mean | 95% CI | Role |
|---|---:|---|---|
| 96 | **-0.07** | **-0.21 to 0.00** | locked test: FAIL |
| 48 (after change 1) | +27.79 | +18.64 to +37.14 | reported: the first change replicates the rule-switch result |
| 72 | -3.86 | -7.21 to -1.14 | reported |

At round 96, 14 of 14 seeds had an advantage of 4 or less (13 exactly 0).

## What this establishes, and what it does not

- **The mechanism worked for the new rule and was unnecessary for the old one.** After change 1 (to a rule Fly B had never seen), advice gave a large teacher-specific lead, as in every earlier rule-switch test. After change 2 (back to the rule Fly B had already learned), Fly B **alone** recovered almost completely within 24 rounds (69.4 of 72), so advice had nothing left to add. This matches the earlier fault-response pilot: advice is not needed when recovery alone is quick.
- **A small cost was seen (reported, not tested).** At 24 rounds after change 2, taught students were about 4 wins behind learning alone, apparently because the strongly taught rule-B habit took slightly longer to unwind. It had disappeared by round 96.
- **What it does not show:** anything about tasks with larger action spaces, noisy outcomes, partial observability, or changes to genuinely new rules each time. The only non-trivial deterministic rules with this cue are A and B, so a second change *to a new rule* could not be tested without changing the task.

Files: per-seed `data/repeated_change/seed_0.json`-`seed_14.json` (seed 9 records its gate failure), implementation check `smoke_seed_20.json`, summary `data/repeated_change_summary.json`. Runner `tools/check_repeated_change.mjs`; analysis `tools/analyze_repeated_change.py`.
