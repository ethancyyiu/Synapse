# Independent KC-based recovery learning: initial gate failed

**.** The [separately locked protocol](practical-transfer-kc-learning-protocol.md) tested whether one Fly A can learn the *synthetic* one-attempt package-recovery task from **measured simulated KC spike activity and its own selected-action rewards**. It used the fixed 42-bucket KC representation from the prior reward-free information study, the unchanged `RecoveryRidge` controller, and the existing 480-attempt budget. The initial Fly A improved but **failed two required learning gates**. By the prospective stop rule, **no independent confirmation, Fly B training, or neural teaching was run**. The earlier failed 42-MBON experiment and successful RPS artifacts remain unchanged.

## Locked implementation and matched comparison

The initial neural identity was **19402**. The KC arm and archived raw-observation arm shared the original preselected training and held-out case/outcome tapes (`20261109111` / `20261109112`) and controller RNG seed `20261109118`. Each KC attempt drove the three existing VPN lanes at `40+110×(offset,slip,resistance)` Hz over 0–240 ms, reset ordinary counts at 50 ms, and read actual KC spikes at 240 ms. The **2,045 KC counts** were summed into 42 predetermined index-modulo-42 buckets, transformed with `log1p`, and standardized using only the archived 19101 reward-free calibration recordings. The 43-component controller input contained an intercept and these 42 measured features; it never contained the raw telemetry or an analytical reconstruction. Full neural state reset each attempt; all KC→MBON multipliers remained one.

The synthetic case generator and all outcome probabilities/costs were unchanged. The first 288 actions followed the outcome-independent 12-per-octant/action coverage schedule; the next 192 used the existing ridge/Thompson exploration. Only the selected action's observed success and net reward updated its controller arm. Checkpoints at 0/288/480 reused one independent 120-case held-out neural tape with exploration and learning disabled. The raw learner is a matched observable-input reference, **not** a second fly or a teacher.

## Frozen results

Expected net reward uses the known synthetic probabilities **only for after-decision analysis**. Realized recoveries use the fixed held-out potential-outcome tape and can differ from expected reward.

| Training attempts | KC expected net reward | KC recoveries /120 | KC A0/A1/A2 | Raw expected net reward | Raw recoveries /120 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .3912 | 71 | 120/0/0 | .3912 | 71 |
| 288 | .5176 | 96 | 23/55/42 | .6366 | 107 |
| 480 | **.5711** | **94** | **29/52/39** | **.6412** | **108** |

The KC policy gained **+.1799** over its own initial frozen expected reward and ended **+.0667** above the best constant action (.5044), but closed only **46.6%** of the positive oracle-minus-constant gap (.6477−.5044). Its final expected reward was **.0701 below** the matched raw learner. All three recovery actions occurred at least 12 times in frozen evaluation. Final KC held-out delay was **539 modeled seconds** with **39 escalations**; raw used **571** and **51**, but the KC arm had fewer realized successes. These synthetic costs are descriptive and cannot establish a better recovery policy.

Final KC results across the eight observable octants (15 held-out cases each):

| Octant | Recoveries | Mean expected net reward | A0/A1/A2 |
| ---: | ---: | ---: | ---: |
| 0 | 11 | .5895 | 7/7/1 |
| 1 | 11 | .5050 | 3/3/9 |
| 2 | 11 | .7114 | 1/13/1 |
| 3 | 13 | .5796 | 0/5/10 |
| 4 | 14 | .6265 | 9/3/3 |
| 5 | 12 | .4724 | 5/4/6 |
| 6 | 10 | .5564 | 2/9/4 |
| 7 | 12 | .5281 | 2/8/5 |

## Training coverage, failures, and integrity

| Training phase | KC actions A0/A1/A2 | KC successes | KC failures | Delay seconds | Escalations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forced attempts 1–288 | 96/96/96 | 205 | 83 | 1,248 | 96 |
| Reward-guided attempts 289–480 | 47/80/65 | 161 | 31 | 869 | 65 |
| **All 480** | **143/176/161** | **366** | **114** | **2,117** | **161** |

The matched raw learner had **376 successes, 104 failures, 2,156 modeled delay seconds, and 170 escalations** across the same 480 cases. The forced phase is identical between arms because its actions and outcome tape were fixed. The difference arises during reward-guided decisions, but neural feature variability, normalization, feature dimension, and the resulting feedback trajectory are not separately randomized factors here. The experiment therefore does not isolate which one explains the performance gap.

Mean KC recruitment was **33.0** active neurons per training observation (range **3–110**); the fixed normalized bucket features ranged **−.592 to +2.117**. Stored rows include actual KC/VPN counts, requested drives, timing, neural keys, inputs, choices, selected rewards, and checkpoints. A keyed replay exactly reproduced the first train and held-out neural probes. Controller parameters changed, only selected-action counts advanced, all KC→MBON multipliers remained one, and each frozen checkpoint preserved its controller snapshot. The initial raw tape and archived raw checkpoints reproduced exactly. Saved-row checks independently recomputed the reward of every selected training and frozen action. No raw telemetry, correct-action label, hidden outcome probability, or future reward entered the KC controller.

## Gate ledger and stop decision

| Prospectively locked criterion | Outcome |
| --- | --- |
| Correctness, fixed KC extraction, 480/120 cases, 288 coverage, deterministic replay, unchanged neural weights | **Pass** |
| Final KC expected reward gain ≥.05 over its initial checkpoint | **Pass:** +.1799 |
| Final KC expected reward ≥.05 over best constant | **Pass:** +.0667 |
| Close ≥50% of oracle-minus-constant gap | **Fail:** 46.6% |
| ≥12/120 frozen uses of every action | **Pass:** 29/52/39 |
| Final KC expected reward no more than .05 below matched raw | **Fail:** −.0701 |
| Controller changes; frozen copies and KC→MBON weights unchanged | **Pass** |

**The joint initial learning gate failed.** The confirmation seed **19403** and its streams were declared in the protocol before this initial result, but were **not run**, as required. Analytical KC decodability did not ensure the reward-trained controller would match the raw-observable learner within the locked margin. This result does not prove KC input cannot support learning with other methods or budgets, and it does not establish practical robot performance. There is insufficient evidence to justify a package-recovery teacher–student experiment under this locked architecture. The single next step for this branch is **stop and preserve the failed independent-learning gate**; any different KC learner would require a new question and prospective protocol.

## Artifact and reproduction

The [write-once initial artifact](../data/practical_transfer_kc_learning_initial.json) contains all case tapes, 480 measured neural training rows, 120 measured held-out neural rows, controller training rows and snapshots, three complete frozen evaluations, octant summaries, calibration constants, and all gate values. Its locked-protocol SHA-256 is `2deaab67a5425f46dd741a39c0d8aa182d8c421cbbc9e40c91cb2b335da7a45d`. There is **no confirmation artifact**.

From the repository root: `node --check tools/check_practical_transfer_kc_learning.mjs`; `node tools/test_practical_transfer_redesign_controller.mjs`; `node tools/check_practical_transfer_kc_learning.mjs preflight`. To reproduce the full initial simulation, run `node tools/check_practical_transfer_kc_learning.mjs initial` **only in a copy without the write-once artifact**. The runner refuses overwrite. The saved protocol hash, 480/120 row counts, 43-component measured feature vectors, chosen-action rewards, frozen rows, and absence of confirmation were verified read-only. No Git, demo, RPS, teacher–student, or neural-dynamics change occurred.
