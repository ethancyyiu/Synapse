# Package-recovery sensory validation: independent streams

** result: the locked pooled-decoder gate failed.** This reward-free study used the [prospectively locked protocol](package-recovery-sensory-validation-protocol.md) and the [write-once 180-probe artifact](../data/package_recovery_sensory_validation.json). No teacher or student was trained; the prior [practical-transfer teacher failure](practical-transfer-report.md) remains unchanged. The protocol SHA-256 recorded in the artifact is `3b8a08584832e52e243fe6338531ec6fab302aa4ceaecbf08efae96a991bd23f`.

## Method and gates

Calibration neural seed **17101** supplied six reward-free recordings per sensory cue to construct the three nearest centroids of the existing standardized, pooled MBON features. Six additional recordings per cue were a disjoint internal check. Validation seeds **17102** and **17103** each supplied 72 independent training-style streams, 24 per cue in a prespecified shuffled order. Every probe reset the neural state, drove the original cue VPN lane at 150 Hz over 0–240 ms, reset spike counts at 50 ms, and measured at 240 ms. The decoder chose the minimum squared-Euclidean-distance centroid; no outcome or reward entered its fit. All VPN counts, active KC identities/counts, 42 individual MBON counts, three pooled rates, three distances and margin, and compact reset/terminal state summaries are in the artifact. The older 17001 centroids were a **secondary comparator only**.

The locked gate required valid recordings and unchanged weights, intended VPN lane uniquely largest and positive on **24/24** trials per cue and seed, and for **each** validation seed at least **63/72** total, **20/24** per cue, and no more than **4/24** C0→C2 errors. The recording and VPN gates passed. The pooled-decoder gate failed because seed 17102 recognized C2 only **18/24**. No gate, centroid, feature, or threshold was changed after evaluation.

## Independent results

Confusion matrices have intended C0/C1/C2 as rows and predicted C0/C1/C2 as columns.

| Recordings / decoder | Matrix | Total | C0 | C1 | C2 | C0→C2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17101 disjoint internal check / new centroids | `[[5,0,1],[0,6,0],[1,0,5]]` | 16/18 | 5/6 | 6/6 | 5/6 | 1/6 |
| 17102 independent / new centroids **primary** | `[[22,2,0],[0,24,0],[6,0,18]]` | 64/72 | 22/24 | 24/24 | **18/24** | 0/24 |
| 17103 independent / new centroids **primary** | `[[23,1,0],[2,22,0],[3,0,21]]` | 66/72 | 23/24 | 22/24 | 21/24 | 0/24 |
| 17102 independent / archived 17001 centroids **secondary** | `[[22,2,0],[0,24,0],[3,0,21]]` | 67/72 | 22/24 | 24/24 | 21/24 | 0/24 |
| 17103 independent / archived 17001 centroids **secondary** | `[[23,1,0],[2,22,0],[1,0,23]]` | 68/72 | 23/24 | 22/24 | 23/24 | 0/24 |

The archived centroids happen to meet the numerical cutoffs on these streams, but selecting them from this comparison would be post-evaluation tuning. They cannot rescue the prespecified primary failure. The earlier teacher 17001's **30/47 C0→C2 training errors** did **not** recur here: both new validation streams had 0/24 C0→C2. Instead the new centroids made **6/24** and **3/24** C2→C0 errors. Thus the exact old direction is stream/calibration dependent, while independent pooled decoding is not uniformly reliable under this gate.

## Where the recorded responses differ

- **VPN:** The intended lane had the unique positive spike maximum on all 144 validation probes, including every error. Mean intended C0 lane counts were roughly 420; intended C2 counts roughly 395. In the six misclassified C2 trials for 17102, the intended lane averaged 386.5 spikes versus 398.9 in its correctly classified C2 trials; for 17103 the corresponding means were 379 versus 396.9. These modest differences do not establish a stimulation defect.
- **KC:** C0 and C2 active-neuron identities differed descriptively. Mean pairwise active-set Jaccard similarity was **0.764/0.778** within C0, **0.684/0.666** within C2, and only **0.078/0.081** between C0 and C2 for seeds 17102/17103. Misclassified C2 probes recruited fewer active KCs on average (**5.2/4.7**) than correctly recognized C2 probes (**7.3/6.7**). These are population descriptions, not validated KC decoding accuracy.
- **MBON:** The 42 individual counts and three pooled rates both differed between C0 and C2. Descriptive mean-separation/within-class-RMS ratios for 42 individual counts versus pooled rates were **3.303 versus 3.153** for 17102 and **4.060 versus 4.106** for 17103. No consistent advantage for the individual counts was shown, and these dimension-dependent ratios are not classifier comparisons. Misclassified C2 pooled means were **(0.536, 0.131, 0.857)** versus correctly classified C2 **(0.750, 0.377, 1.250)** for 17102; the corresponding 17103 values were **(0.548, 0.119, 0.929)** versus **(0.718, 0.367, 1.272)**. Lower responses on these error trials moved the pooled vector toward C0. The saved distance margins are generally positive, so these errors need not be exact centroid ties.
- **Calibration versus validation:** For C0 the calibration pooled mean **(0.452, 0.071, 0.345)** was broadly similar to validation means **(0.426, 0.092, 0.304)** and **(0.381, 0.068, 0.307)**. For C2 the calibration mean **(0.774, 0.476, 1.286)** exceeded validation means **(0.696, 0.315, 1.152)** and **(0.696, 0.336, 1.229)**. This is a descriptive calibration-to-stream difference consistent with C2→C0 errors, not proof of its cause. The much higher C0 rates seen in the original 17001 training stream were not reproduced here.

The data establish separated VPN delivery, distinct KC population patterns, and class-related individual and pooled MBON responses. They do **not** locate an irrecoverable information loss at KC, individual MBON, or pooling, because no independent classifier was locked and tested at those intermediate layers. They also do not separate calibration-sample variation, neural response variation, and stream distribution effects causally. A new decoder is **not justified for installation** from these descriptive comparisons.

## Decision and reproducibility

Stop this package-recovery decoder branch before HackPrinceton. The single next step is to retain the failed sensory and teacher gates in the research record and use the independently audited RPS evidence for the demonstration. Any future decoder proposal would require a separate prospective calibration and independent validation protocol; these results do not support choosing a replacement now. No package-recovery teacher–student experiment follows.

Focused checks: `node --check tools/check_package_recovery_sensory_validation.mjs`; protocol SHA-256 matched the saved artifact; the original practical-transfer artifact hash remained `A9B63D2580F86A44D595091130F769A6F6058311CDB71D1503A945CDCC0755E7`. To reproduce the **one** frozen batch, use a copy without the write-once result file and run `node tools/check_package_recovery_sensory_validation.mjs`. Do not rerun in this repository: the runner refuses overwriting the existing artifact. No training, reward update, Git operation, or demo change occurred.
