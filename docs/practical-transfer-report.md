# Practical transfer: initial synthetic recovery study

** — stopped at the locked teacher prerequisite.** The [protocol](practical-transfer-protocol.md) was saved before the new run. The isolated [Bayesian controller](../tools/practical_transfer_controller.mjs), [runner](../tools/check_practical_transfer.mjs), and [write-once trial artifact](../data/practical_transfer_initial.json) preserve what was tested. No student training, DN experiment, five-arm comparison, or independent confirmation followed the failed gate. Earlier recovery and RPS studies and the HackPrinceton demo remain unchanged.

## Controller and causal scope

The controller receives three standardized, **actually simulated MBON rates**, routes them through a fixed nearest-centroid decoder fitted on reward-free recordings of the *observable* signature, and keeps separate Beta success estimates for each decoded signature/action. It first collects six **own selected-action outcomes** per action within each decoded signature, then uses one posterior-standard-deviation optimism bonus while training. Frozen decisions maximize estimated cost-adjusted net reward without the bonus. No hidden failure cause, unchosen outcome, correct-action label, or population success probability enters its decision/update API. The analytical cause-conditioned expected reward below is used only to score frozen policies. KC→MBON plasticity is frozen; this is a hybrid engineered controller, not original biological dopamine learning. A future student would additionally imitate a preference decoded from measured DN spikes, but this study never reached that stage.

The six-calibration/six-held-out sensory gate passed for both preselected neural seeds: teacher **17001** and reserved student **17002** each routed VPN inputs **6/6 per cue** and recognized MBON signatures **5/6 C0, 6/6 C1, 6/6 C2**. This small gate did not guarantee recognition across the full training/evaluation streams.

## Locked teacher result

| Frozen checkpoint after own-reward cases | New controller expected net reward | Actual successes/36 | Frozen A0/A1/A2 | Same-tape historical LinUCB expected reward |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0930 | 15 | 36/0/0 | +0.0930 |
| 24 | +0.1858 | 17 | 21/15/0 | +0.1283 |
| 48 | +0.1711 | 16 | 25/11/0 | +0.0704 |
| 96 | +0.1711 | 16 | 25/11/0 | +0.1115 |
| 144 | **+0.1711** | **16** | **25/11/0** | **+0.1399** |

The locked teacher required **≥+0.20** final expected reward, **≥+0.10** gain from its own checkpoint 0, **≥22/36** actual recoveries, **≥4** frozen uses of every action, and clearing on **≥60%** of C2 cases. It missed every one of those performance conditions: gain was **+0.0781**, and C2 clearing was **0/15**. Controller parameters changed; all KC→MBON multipliers remained one. The separate source-preference quality gate was not run because the teacher had already failed.

Training and frozen performance differ. Across 144 training attempts the new controller played A0/A1/A2 **69/49/26**, had **61 successes and 83 failed attempts**, and incurred **542 delay seconds** and **26 escalations**. The same-tape unchanged LinUCB core had **62 successes, 82 failures**, 496 delay seconds, 25 escalations, and final expected reward +0.1399. The new controller's 24/48/96 early mean expected reward was **+0.1760**, versus **+0.1034** for LinUCB, but it did **not** reduce failed attempts or produce a passing final policy. This is one matched new-tape comparison, not a revision of LinUCB's earlier failed gates or proof of better sample efficiency.

## What the saved rows explain

The fixed decoder recognized **17/47** true C0 training inputs as C0; **30/47** were routed to C2. On the same frozen evaluation tape, it routed only **3/9** C0 cases correctly, although all **15/15** C2 cases were routed correctly. The earlier six-probe C0 gate had been **5/6**. Neural state was reset for each case, so this pattern shows substantial stream-to-stream MBON feature variability or calibration mismatch, not evidence of carryover during training. The protocol's small sensory gate was too weak to reveal that variability on this seed. It remains a valid predeclared pass; the teacher-learning gate remains a valid failure.

The C2 posterior head received **14** clearing-action outcomes, but only **six** were from genuine C2 signature cases; the other eight were misrouted C0 cases. Those 14 attempts yielded seven successes overall. For final correctly decoded C2 inputs, estimated frozen net scores were **A0 +0.0067, A1 −0.3450, A2 −0.1400**, so it chose A0 on **15/15** C2 cases. This directly supports a learned-value and sensory-routing limitation. The archive cannot isolate how much of the failed policy was caused by the six-sample calibration, stochastic selected outcomes, or the controller's fixed optimism rule. No parameter was changed after observing this result.

## Decision, integrity, reproduction

The independent Fly A prerequisite failed. Thus whether a trained neural teacher preference would help Fly B recover packages, whether DN messages would decode in this branch, and whether a student would retain a policy are **untested**. There is no independent confirmation and no practical robot evidence. Keep the audited Strategy A/B RPS teaching results as the HackPrinceton research demonstration; this synthetic recovery attempt belongs in the limitations record, not as a successful application claim.

The write-once artifact embeds SHA-256 `b6376c03e266b555dc1d3e98b07486342f2e42a40bcaf22d81aa27f45d7ff188` of the locked protocol and contains calibration/held-out sensory rows, every teacher and matched LinUCB training choice and selected reward, all frozen trial rows, snapshots, and gates. `node --check tools/check_practical_transfer.mjs`, `node --check tools/practical_transfer_controller.mjs`, and `node tools/test_practical_transfer_controller.mjs` passed. To reproduce **in a copy without the saved result file**, run `node tools/check_practical_transfer.mjs initial`; the runner refuses overwrite. Do not run `confirmation` because the initial gate failed.

## Read-only C0 routing diagnosis

This section recomputes routing from the **unchanged** [initial trial artifact](../data/practical_transfer_initial.json) and inspects the existing [runner](../tools/check_practical_transfer.mjs) and [decoder/controller](../tools/practical_transfer_controller.mjs). It does not revise the passed six-probe sensory gate or the failed teacher gate. No simulation, retraining, new seed, or decoder fit was performed.

### Intended signature versus selected head

Each cell is a count across the 144 teacher training attempts. Rows are the *observable* sensor signature that drove the VPN lane; columns are the MBON-decoded policy head. Recomputing the Euclidean squared distances from every saved three-feature vector to the **archived fixed centroids** reproduces all 144 selected heads, including **30/47 C0→C2**.

| Intended signature | Head C0 | Head C1 | Head C2 |
| --- | ---: | ---: | ---: |
| C0 alignment offset | 17 | 0 | **30** |
| C1 slip/rotation | 9 | 49 | 0 |
| C2 high resistance | 0 | 0 | 39 |

The identical frozen evaluation features were generated at all five checkpoints; its independent 36-case tape had C0 **3/9** recognized, with **6/9** sent to C2, C1 **11/12** recognized, and C2 **15/15** recognized. Thus the training confusion is not an artifact of reward updates changing the frozen neural feature vectors.

Misrouting is distributed across training quarters: C0→C2 was **7/10**, **9/11**, **4/10**, and **10/16** in attempts 1–36, 37–72, 73–108, and 109–144. Grouping C0 attempts by preceding training signature gives **11/16** after C0, **9/17** after C1, **9/13** after C2, and **1/1** on the first attempt. These small groups show no unique preceding-cue transition or monotonic training drift. The runner calls `net.resetState()` before each attempt, resets ordinary counts at 50 ms, assigns a new keyed random stream, and samples MBON rates at 240 ms **before** reward/DN activity; neural multipliers remain fixed. The artifact does not store membrane/conductance snapshots or training VPN/KC spike counts, so it cannot independently prove the complete reset state or locate the spike-level origin of the rate variation. No carryover mechanism is established.

### The fixed centroid boundary and calibration mismatch

For true C0 training inputs decoded correctly as C0 (**17**), mean pooled MBON rates were **(0.424, 0.345, 0.340)**. For the **30** decoded as C2, they were **(0.760, 0.600, 0.793)**. All three groups were higher in the misrouted set; no pooled MBON rate was exactly zero in either C0 group. Let `d0` and `d2` be squared Euclidean distances in the fixed standardized feature space to the C0/C2 centroids. The misrouted C0 rows had mean `d0−d2 = +3.538` (range **+0.071 to +10.118**); **23/30** exceeded +1. The correctly routed C0 rows had mean **−1.624**. Most errors are therefore substantial boundary crossings rather than exact ties. The saved action head is precisely the nearest-centroid result; no distance, argmin, feature-standardization, or head-index mismatch was found.

The centroids were fitted on **six reward-free C0 recordings** and six per other cue, with distinct keys and no training or evaluation outcomes. Six further independent C0 probes in the same calibration key family scored **5/6**. Those twelve C0 probes had mean pooled MBON rates **(0.452, 0.083, 0.310)** for calibration and **(0.417, 0.071, 0.286)** for its small held-out check. The **47** C0 training probes instead averaged **(0.638, 0.508, 0.629)**; nine independent frozen C0 probes averaged **(0.587, 0.476, 0.516)**. Calibration/held-out, training, and frozen evaluation use disjoint keyed random streams, with no effective RNG-key overlap in the saved data. The training/evaluation C0 activity is markedly higher, especially in the second and third pooled MBON rates. This demonstrates a **sample/stream mismatch for this fixed decoder**, but the archive does not show whether it is ordinary stochastic neural variability, an unusual six-recording calibration sample, a property of the keyed streams, or an unrecorded spike-level issue. It does not establish a changed biological network state.

### Consequence for actions and value updates

On correctly routed C0 trials, actions A0/A1/A2 occurred **9/4/4**, with **9/17** successes and mean sampled net reward **+0.178**. On misrouted C0 trials, actions were **20/2/8**, with **15/30** successes and mean sampled net reward **+0.135**. These groups have different actions and hidden-cause mixes; their reward difference is descriptive, not an isolated effect of the decoder.

`RecoveryBayesAgent.decide()` passes the MBON-decoded head to `RecoveryBayesCore.choose()`. The later selected-action outcome updates that **same decoded head**, not the intended C0 head. Thus 30 real C0 rewards entered the C2 head. In particular, C2-head A0 received **20 C0 cases with 12 successes** plus **29 true C2 cases with 4 successes**. C2-head A2 received **eight C0 cases with 3 successes** plus **six true C2 cases with 4 successes**. Its final C2-head counts were A0/A1/A2 **49/6/14**. With the fixed action costs, final C2-head frozen values were **+0.0067/−0.3450/−0.1400**, choosing A0 on all 15 held-out true C2 cases. This is direct evidence of cross-context feedback contamination. It does **not** prove that eliminating the contamination alone would satisfy the teacher gate, because the true-C2 selected outcomes were also few and stochastic.

### Verdict and one prospective step

**No implementation bug is confirmed.** Saved feature vectors reproduce the decoded heads exactly; calibration and training use the same pre-action stimulation, MBON pooling, fixed standardization, and nearest-centroid equation; all relevant effective RNG keys are distinct. The observed limitation is that the small, separately keyed sensory calibration did not predict the C0 activity distribution later encountered. Missing training VPN/KC counts, individual MBON counts, trial-start neural snapshots, and a larger independent reward-free probe set prevent a causal account of where the activity shift arose.

**One justified future experiment, not run here:** prospectively lock a *reward-free decoder-validation study* before any further recovery training. Keep the current decoder construction and thresholds fixed, use preselected independent neural identities and separate calibration versus training-style keyed streams, and record VPN, KC, pooled/individual MBON activity plus nearest-centroid margins on a substantially larger balanced cue probe set. Predefine a per-cue held-out recognition floor and stop if C0 fails it. This directly tests whether the six-probe gate generalizes across the stream families without retuning the failed experiment. The present failed teacher result remains final.
