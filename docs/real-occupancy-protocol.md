# Real sensor occupancy: first feasibility gate

**Locked before model evaluation.** This exploratory benchmark uses recorded room sensors, separate from the synthetic package-recovery failures and the RPS studies. It tests raw-data learnability first. A pass alone does not authorize neural simulation, Fly A/B training, or teaching.

## Task and source

The source is the [UCI Room Occupancy Estimation dataset](https://archive.ics.uci.edu/dataset/864/room%2Boccupancy%2Bestimation), DOI `10.24432/C5P605`, licensed CC BY 4.0. The local adapted file is `data/external/occupancy_sensor_anonymized.csv`; it retains the sensor features, occupancy labels, and fixed partition labels. The dataset records measurements from one instrumented room; it is not a robot or HVAC intervention dataset.

The three target categories are **empty** (0 people), **partly occupied** (1 or 2), and **full** (3). This grouping is fixed before evaluation. Classification correctness supplies the prospective reward; the correct category must never be an input to a fly or its controller before a decision. This study does not measure energy savings, ventilation quality, or deployment performance.

Use exactly three readings: `S5_CO2`, the mean sound reading `mean_S1_to_S4_Sound`, and summed PIR reading `S6_PIR_plus_S7_PIR`. Other sensor values and the occupancy label are not predictive inputs. Scale each reading to `[0,1]` using its minimum and maximum on training rows only, clipping later values. A zero training range is an invalid task and stops the run.

## Fixed partitions

- The training partition contains the calibration rows. Select exactly 96 rows from each grouped category by sorting source order within that category and taking index `floor((i + 0.5) * N / 96)` for `i=0..95`. No seed or random sampling is involved.
- The held-out partition is evaluated in full. The saved partition contains 703 empty, 93 partly occupied, and 201 full rows. Records marked excluded are not used for training, scaling, or evaluation.
- The held-out partly occupied cases contain two people, not one; this is a stated generalization limit. The benchmark uses one room and does not test transfer to a new site.

## Fixed raw-information reference and gates

Fit a **5-nearest-neighbor** reference on the 288 selected labeled training rows and their three scaled features. Use squared Euclidean distance, sort ties by source order, choose the most frequent neighbor class, and resolve a class-vote tie toward the lowest class index. This analytical classifier is a raw-information check; it is not installed in Fly A or B and does not receive hidden state beyond the post-observation training labels. No model fitting or threshold selection uses held-out records.

Report the full held-out confusion matrix, per-class recall, balanced accuracy (mean of three recalls), overall accuracy, and the 1/3 balanced constant-class baseline. **Pass only if** there are at least 50 held-out rows per class, balanced accuracy is at least **0.70**, and each class recall is at least **0.60**. Otherwise stop this task without neural simulation or reward training. Preserve any failure; do not change features, category grouping, partitions, classifier, or gates after seeing results. Record every held-out prediction in the write-once JSON artifact with protocol and source hashes.

## Conditional next work

A raw pass would justify a **separately locked** reward-free neural sensory check using these same three scaled readings on existing VPN lanes, followed only conditionally by independent Fly A reward learning and a matched teacher-student study. Each stage needs its own training/held-out separation and stop gate before simulation. Earlier RPS and practical-transfer results remain untouched.
