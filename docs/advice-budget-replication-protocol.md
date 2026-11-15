# Study 5: fresh-seed replication of the 36-message result

Locked before any fresh-seed run. This is a replication, not a method change. It re-runs the 36-message condition of `advice-budget-followup-protocol.md` on seeds no study has used. The follow-up's files, the earlier 12-message study, and every Science/banking artifact stay unchanged.

## What is being replicated

The follow-up found that 36 spaced trained messages over 48 training rounds met its in-sample recovery criterion on seeds 100-114: mean 64.47/72, 13/15 students at or above 58/72, paired lead over the stronger of matched random and shuffled advice +34.60 (95% CI +23.13 to +45.87). Those seeds were already used by the preceding 12-message study, so that result has never been checked on held-out seeds.

**Hypothesis.** On 15 new seeds, 36 spaced trained messages again give mean recovery (mean at least 58/72, at least 12/15 students at or above 58/72) and a teacher-specific benefit (95% paired lower bound over the stronger matched control above 4 wins).

## Fixed design, unchanged from the follow-up

Same connectome, teacher training (12 Strategy B episodes, 24 rounds each), teacher gate (at least 58/72 on three 24-round frozen episodes), student architecture and learning rule, four-episode Strategy A history, 48 post-switch training rounds with 46 observed-cue opportunities, two-neuron DN spike code with measured decoding, own-action rewards on every round, and teacher-disconnected 72-game frozen checks at rounds 0, 24 and 48. All seed formulas are the follow-up's, applied to the new indices (teacher `26001 + 2s`, student `26002 + 2s`, and the same opponent, switch, key and shuffle formulas). The 36- and 46-message schedules are the follow-up's: observed-cue ordinals `floor((i + 0.5) * 46 / B)`.

Arms: `alone` (0 messages), `trained36`, `random36` (random actions at the trained36 times, generated exactly as in the follow-up), `shuffled36` (the trained36 actions Fisher-Yates-shuffled across the same times with the follow-up's seed formula), and `trained46` (reference). Random and shuffled 46-message arms are dropped to save run time; they were not part of the 36-message criterion.

Runner: `tools/check_advice_budget_replication.mjs`, a copy of the follow-up runner restricted to these arms. Write-once per seed in `data/advice_budget_replication/`.

## Seeds

- **Implementation check (not part of the result):** seed index 100. The runner must reproduce the follow-up's saved `alone`, `trained36`, `random36`, `shuffled36` and `trained46` scores for seed 100 exactly at rounds 0, 24 and 48, or it stops. Saved as `smoke_seed_100.json`.
- **Final:** seed indices 200-214 (teacher seeds 26401-26429, student seeds 26402-26430). No study has used them (fly studies used 0-17; the advice-budget studies used 98-114). No seed is replaced. A teacher that fails its gate counts as a teacher-readiness failure and as a non-recovered student.

## Fixed analysis

`tools/analyze_advice_budget_replication.py`, written before the final runs, with the follow-up's bootstrap method (10,000 paired resamples, percentile interval) and new fixed seed `2026111520`.

**REPLICATED** only if all hold:
1. Integrity: at least 10 of 15 teachers pass the gate, and every completed seed decodes all 36 (and 46) scheduled messages.
2. Mean recovery: mean `trained36` at round 48 is at least 58/72 over completed seeds, and at least 12 of the 15 final seeds reach 58/72.
3. Teacher-specific benefit: the 95% paired lower bound of `trained36 - max(random36, shuffled36)` exceeds 4 wins.

**PARTIAL** if 1 holds and exactly one of 2 and 3 holds. **NOT REPLICATED** otherwise.

**Reported, not tested:** means of every arm at 24 and 48 rounds; `trained36 - random36` and `trained36 - alone` with intervals; `trained46` recovery; per-seed scores; actual sent and decoded counts; teacher gate scores and failures.

Nothing changes after fresh results are seen. A failure is reported as a failure, and possible causes are examined only afterwards, in a clearly separate section.

## Limits

Same simulator and task as the follow-up, so a pass shows the 36-message result holds on new seeds of this simulation, not in other tasks, real brains or banking. It still does not identify the exact minimum budget; 25-35 messages remain untested.
