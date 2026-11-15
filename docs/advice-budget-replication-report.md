# Study 5 result: the 36-message result partly replicates on fresh seeds

**Verdict under the locked rule: PARTIAL.** The teacher-specific benefit replicated clearly. Mean recovery passed, but too few individual students recovered: 9 of 15, where the rule needed 12. [Protocol](advice-budget-replication-protocol.md), locked before any fresh-seed run.

Simulated Rock-Paper-Scissors only; not biological or banking evidence.

## What was run

- Exact copy of the advice-budget follow-up's runner, restricted to five arms: no advice, 36 trained messages, 36 matched random, 36 shuffled, and 46 trained (reference). Same teacher training, gate, student, learning rule, spike code, 48 training rounds, 58/72 threshold and frozen evaluation.
- **Implementation check:** on archived seed 100 the new runner reproduced the follow-up's saved scores exactly at rounds 0, 24 and 48 (excluded from the result).
- **15 fresh seeds (200-214)**, never used by any study. All 15 teachers passed their gate (59-69 of 72). Every completed seed decoded all 36 (and 46) scheduled messages from measured spikes. No seed failed or was replaced.

## Results at round 48 (frozen, teacher disconnected; wins out of 72)

| Arm | Mean | Students at or above 58/72 |
|---|---:|---:|
| No advice | 13.07 | 0/15 |
| 36 random | 25.47 | 0/15 |
| 36 shuffled | 25.60 | 0/15 |
| **36 trained** | **60.20** | **9/15** |
| 46 trained (reference) | 65.73 | 15/15 |

Paired differences (95% percentile bootstrap, 10,000 resamples, seed 2026111520):

| Contrast | Mean | 95% CI |
|---|---:|---|
| 36 trained minus stronger of random/shuffled (locked test) | **+26.47** | **+18.33 to +35.53** |
| 36 trained minus random | +34.73 | +25.40 to +45.33 |
| 36 trained minus no advice | +47.13 | +37.20 to +56.13 |
| 46 trained minus no advice | +52.67 | +43.07 to +61.27 |

At round 24: no advice 2.67, 36 trained 43.73, 46 trained 48.73.

### Locked criteria

| Criterion | Required | Result | Holds |
|---|---|---|---|
| Integrity | at least 10 teachers pass; all messages decode | 15/15; 36/36 and 46/46 | yes |
| Mean recovery | mean at least 58 **and** at least 12/15 at 58 | mean 60.20; **9/15** | **no** |
| Teacher-specific benefit | lower bound above +4 | +18.33 | yes |

### Per-seed frozen wins at round 48

| Seed | Teacher gate | None | 36 trained | 36 random | 36 shuffled | 46 trained |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 69 | 24 | 46 | 22 | 44 | 64 |
| 201 | 67 | 1 | 47 | 24 | 24 | 69 |
| 202 | 61 | 2 | 68 | 1 | 26 | 68 |
| 203 | 66 | 2 | 65 | 49 | 26 | 70 |
| 204 | 59 | 47 | 50 | 24 | 48 | 58 |
| 205 | 67 | 0 | 58 | 0 | 23 | 62 |
| 206 | 68 | 2 | 55 | 26 | 2 | 58 |
| 207 | 68 | 1 | 61 | 46 | 46 | 64 |
| 208 | 66 | 44 | 70 | 47 | 40 | 70 |
| 209 | 68 | 2 | 56 | 49 | 26 | 65 |
| 210 | 64 | 45 | 68 | 45 | 0 | 69 |
| 211 | 65 | 2 | 69 | 2 | 25 | 69 |
| 212 | 64 | 22 | 64 | 22 | 29 | 66 |
| 213 | 64 | 1 | 56 | 24 | 24 | 64 |
| 214 | 64 | 1 | 70 | 1 | 1 | 70 |

## What this establishes, and what it does not

- **Established (on fresh seeds):** 36 trained messages give a large benefit that comes from the teacher's content, not from receiving messages at all. Random and shuffled advice at the same moments did far worse.
- **Not established:** that 36 messages *reliably* bring a student to recovery by round 48. Over the two cohorts, 36 messages recovered 13/15 (follow-up, reused seeds) and 9/15 (here, fresh seeds). 36 looks like a borderline budget for this task; 46 recovered 15/15 here.
- The exact minimum budget is still unknown (25-35 never tested).

## Looked at only after the locked result (not part of the test)

Five of the six students that missed 58/72 scored 46-56, so most were near the threshold rather than far from it. The weakest teacher (seed 204, 59/72) went with one of the lowest trained scores (50). These are observations for future work, not explanations: with 15 seeds they cannot separate teacher quality from chance.

Files: per-seed `data/advice_budget_replication/seed_200.json`-`seed_214.json`, implementation check `smoke_seed_100.json`, summary `data/advice_budget_replication_summary.json`. Runner `tools/check_advice_budget_replication.mjs`; analysis `tools/analyze_advice_budget_replication.py`. No earlier file was changed.
