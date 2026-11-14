# Real-data advice after a change, v2: fraud follow-up and sepsis (15 seeds each)

**Locked before any v2 run, and before any model is fitted on the sepsis data.** Runner: `tools/run_real_data_advice_v2.py`; its SHA-256 and this file's go into every result file. Everything not stated here is unchanged from `docs/real-data-advice-protocol.md` (v1).

## What happened in v1, and what to hold against v2

v1 ran on fraud only and **failed**: advice made the student worse in 14 of 15 seeds, mean `taught - best control` -0.118 (95% CI -0.178 to -0.064). It stays reported as failed (`data/real_data_advice/fraud/`, `fraud_summary.json`). Sepsis was never run under v1; v2 replaces it.

Reading the v1 results showed three design faults. All three changes below were made **after seeing the v1 fraud results**, so the fraud part of v2 is a follow-up designed with knowledge of a failure, the same situation as the fault-response cohort (`docs/fault-response-cohort-protocol.md`). Sepsis has had no run of any kind.

1. **No teacher check.** The v1 teacher scored 0.36 average precision while the student it advised scored 0.74 alone; most of its advice was wrong. Every fly study here gates the teacher before it teaches; v1 left that out.
2. **No room to help.** Even the true labels for all 4,500 advised cases moved the student from 0.742 to 0.749. With about 8 frauds among them, no teacher could have cleared the +0.02 bar.
3. **Wrong byte count.** v1 says the advice was 625 bytes. The advice covered the 4,500 unlabelled cases, so it was 4,500 bits, 562 bytes. The v1 result is unaffected.

## Changes from v1

- **Teacher picks its own settings, on its own data.** Its private slice is split by unit (transaction or patient) 75/25 with `numpy.random.default_rng(5000 + seed)`. On the 75% part, 3-fold `GroupKFold` by unit chooses, by mean average precision, one of four `HistGradientBoostingClassifier` settings: `learning_rate` in {0.05, 0.1} × `max_leaf_nodes` in {15, 31}, with `max_iter=200`, `l2_regularization=1.0`, `early_stopping=False`, `random_state=seed`. The chosen setting is fitted on the whole 75% part. The out-of-date teacher follows the same procedure on its old-data sample.
- **Teacher gate, per seed.** On the 25% part of its own slice, the teacher's average precision must be at least the old-only student's average precision on the same cases **+ 0.02**. A seed that fails stops there and is reported; it is not replaced.
- **More advised cases.** `N = 25,000` cases from the stream (sepsis: patient-hours of stream patients), of which `L = 500` carry a true label; the other **24,500 get one advice bit each: 3,062 bytes**.
- **Headroom check, per dataset.** Mean `fullLabels - alone` must be ≥ 0.02. If not, the dataset cannot show an advice effect and the main test is reported as **uninformative**, not passed.
- **New seeds**: 100-114 (v1 used 0-14). All rng formulas from v1 use these seed numbers.

## Locked tests, per dataset

1. **Change**: mean `newSiteStudent - oldOnlyStudent` ≥ 0.02 (as v1).
2. **Headroom**: mean `fullLabels - alone` ≥ 0.02, over seeds that pass the teacher gate.
3. **Main test**: over seeds that pass the teacher gate, `advantage = taught - max(alone, shuffled, random)`. **Pass:** at least 10 seeds pass the teacher gate, tests 1 and 2 pass, and the 95% percentile bootstrap lower bound of the mean advantage is > 0.02 (10,000 resamples, rng `2026111411` fraud, `2026111412` sepsis).

Reported, not tested: as v1, plus the teacher gate margin per seed and the chosen teacher settings.

No seed is replaced and nothing changes after results are seen. Any failure is reported on the page as a failure, and v1's fraud failure is reported alongside v2.

## Limits

As v1. In addition: fraud v2 is the second test on the same two days of data, with its design shaped by the first test's failure.
