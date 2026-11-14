# Real-data advice after a change: fraud and sepsis (15 seeds each)

**Locked before any model is fitted on either dataset.** The only thing seen before locking was each dataset's column list and size. The runner is `tools/run_real_data_advice.py`; its SHA-256 and this file's are written into every result file.

## Why this test exists

The Research tab describes fraud detection and hospital diagnostics as places where "a partner that has already adapted could help a differently built one catch up by sending a few bytes of advice." Nothing on the page tested that on real data. This study does, with ordinary machine-learning models on two public datasets. **It does not use the fly connectome, spikes, or either fly.** A pass would show the advice idea, not the neural channel, carries over to real fraud and clinical data.

## Data

- **Fraud.** Credit-card transactions by European cardholders over two days, September 2013 (Worldline and ULB; Dal Pozzolo et al. 2015), OpenML dataset 1597. 284,807 transactions, 30 features (`Time`, PCA components `V1`-`V28`, `Amount`), label `Class`. The **change** is time: the old period is day 1 (`Time < 86400`), the new period is day 2. `Time` itself is not a feature.
- **Sepsis.** PhysioNet / Computing in Cardiology Challenge 2019, hourly ICU records from two separate US hospital systems (set A, set B), CC BY 4.0. Label `SepsisLabel` per hour. The **change** is the hospital: the old site is hospital system A, the new site is hospital system B. Values are carried forward within a patient; remaining gaps stay missing. Features: the 34 vital and lab columns, `Age`, `Gender`, `HospAdmTime`, `ICULOS`. `Unit1`, `Unit2` and the patient id are not features. All splits are by patient, so no patient's hours are in two parts.

## Roles

- **Teacher (the "partner that has already adapted").** A gradient-boosted tree model (`HistGradientBoostingClassifier`, `max_iter=200`, `learning_rate=0.1`, `random_state=seed`), trained only on its own private slice of new-period/new-site data. It never shares weights or data. Its only message is one bit per case: `1` ("I would flag this") if its predicted probability is ≥ 0.5, else `0`.
- **Student (the "differently built one").** A logistic regression (`C=1.0`, `lbfgs`, `max_iter=2000`) on median-imputed, standardised features plus a missing-value indicator per feature; imputation and scaling are fitted on its own old-period/old-site data. Its structure cannot accept the teacher's trees.
- **Out-of-date teacher.** The same tree model trained on an old-period/old-site sample of the same size as the current teacher's slice.

## Per seed (seeds 0-14)

New-period/new-site rows (fraud) or patients (sepsis) are shuffled with `numpy.random.default_rng(1000 + seed)` and cut into: **teacher's private slice 40%**, **student's stream 20%**, **held-out test 40%**. From the stream, exactly **N = 5,000 cases** are drawn (sepsis: patient-hours from stream patients), and **L = 500** of those come with a true label (labels after a change are scarce or late; fraud chargebacks take weeks). Advice covers all 5,000 cases: 5,000 bits, **625 bytes**.

The student is refitted once after the change on: its old-period/old-site data with total weight 1,000 (each row weighted `1000 / n_old`), plus the cases below with weight 1 each.

| Arm | Cases added after the change |
|---|---|
| `alone` | the 500 labelled cases |
| `taught` | the 500 labelled cases, plus the other 4,500 cases labelled with the current teacher's bit |
| `shuffled` | as `taught`, but the 4,500 bits randomly permuted among the cases (rng `2000 + seed`) |
| `random` | as `taught`, but bits drawn at random at the teacher's flag rate (rng `3000 + seed`) |
| `stale` | as `taught`, but bits from the out-of-date teacher |
| `fullLabels` (reference) | all 5,000 cases with their true labels |

Fraud old-period data for the student and the out-of-date teacher: all day-1 transactions (the out-of-date teacher uses a random day-1 sample of the teacher-slice size, rng `4000 + seed`). Sepsis: all hospital-A patients, and the out-of-date teacher uses a random hospital-A patient sample of the teacher-slice patient count (rng `4000 + seed`).

## Measure

Average precision (area under the precision-recall curve) on the held-out test cases, for each arm, the current teacher, and the out-of-date teacher. Also reported: recall when the student flags its top 1% of test cases.

## Locked tests, per dataset

1. **Is there a change?** `newSiteStudent - oldOnlyStudent`, where `oldOnlyStudent` is the student fitted on old data alone and `newSiteStudent` is the student fitted on the teacher's slice with true labels. Must have a mean ≥ 0.02. If it fails, the dataset shows no change worth recovering from, and test 2 is reported but cannot support "after a change".
2. **Main test.** `advantage = taught - max(alone, shuffled, random)` per seed. **Pass:** 15 seeds complete and the 95% percentile bootstrap lower bound of the mean is > 0.02 (10,000 resamples, rng `2026111401` fraud, `2026111402` sepsis).

**Reported, not tested:** `taught - stale`; `fullLabels - taught`; teacher and out-of-date teacher scores; mean of every arm; recall at top 1%; seeds with `advantage <= 0`.

No seed is replaced, and nothing above changes after results are seen. A failure on either dataset is reported on the page as a failure.

## Limits

Two days of one card processor's data and two hospital systems: a real change, but one each. The advice bits are given as training labels, which is a standard way to learn from a teacher's outputs (Li & Wang 2019); this is not the engineered imitation update used by Fly B. Labels for 500 cases is a chosen setting, not a measured real-world label delay. No hospital, bank, or deployment was involved.
