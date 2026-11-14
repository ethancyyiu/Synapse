# Synthetic package-recovery learnability diagnostic

**: valid new study; neural information passed, independent learning failed.** The [separately locked protocol](package-recovery-learnability-protocol.md), [write-once trial artifact](../data/package_recovery_learnability.json), and [isolated runner](../tools/check_package_recovery_learnability.mjs) preserve the two predetermined seeds and every gate. Earlier original and LinUCB teacher-gate failures remain unchanged. This is a stipulated one-attempt simulator, not robot evidence.

## Archived failures and available information

The original 14001 teacher ended at **0.1943** frozen cause-conditioned expected net reward, **22/36** realized recoveries, and **20/16/0** A0/A1/A2 choices; its locked all-action teacher gate failed. The matched 15001 LinUCB run had **68 versus 82** failed training attempts and **0.2101 versus 0.0876** mean sampled net reward. Its final frozen expected reward was **0.2114 versus 0.1916**, only **+0.0198**; it recovered **19/36** cases and selected A2 on **7/12** C2 cases. All three original LinUCB teacher requirements that missed remain failed.

The synthetic generator emits **one categorical signature**, not a raw force/vibration time series. The controller sees this signature either directly in the raw control or as three standardized MBON rates produced by VPN→KC→MBON simulation. It never receives hidden cause or unchosen outcomes. After choosing A0 regrip, A1 stabilize, or A2 clear/escalate, it receives only that action's sampled success and cost-inclusive reward. An external LinUCB readout learns from those rewards; KC→MBON synapses remain frozen.

## Information available for decisions

From the fixed equiprobable-cause model, the raw-signature Bayes action is **A0/A1/A2** for C0/C1/C2. Its population expected net reward is **+0.2613 per attempt**. The hidden-cause oracle reaches **+0.4650**, an unavailable gain of **+0.2038**. Because neural activity is generated from the signature, the raw-signature optimum bounds an MBON-only policy under this model. These are model-known reference policies, **not policies learned from sampled rewards**.

A single fixed nearest-centroid cue classifier was fitted on each saved training feature tape and applied to disjoint frozen held-out features. It never acted in an experiment. Its decoded-cue action policy is an **attainable analytical reference, not the exact MBON Bayes optimum**:

| Neural seed and tape | MBON cue recognition | MBON-reference expected reward | Raw-signature reference, same cases | Hidden-cause oracle |
| --- | ---: | ---: | ---: | ---: |
| Archived original 14001 | 31/36 | +0.1758 | +0.2418 | +0.4650 |
| Archived LinUCB 15001 | 35/36 | +0.2285 | +0.2616 | +0.4650 |
| New 16001 | **35/36** | **+0.3526** | +0.3688 | +0.4650 |
| New 16002 | **32/36** | **+0.2279** | +0.2663 | +0.4650 |

The new information gate required **≥30/36** correct and **≥+0.16** decoded-policy expected reward for *each* new seed; both passed before reward training. The held-out case mixes differ, so expected-reward rows should be compared **within** a tape, not pooled as if they were matched. The 14001 reference is lower than the trained original policy on that tape; a nearest-centroid decoder is only one imperfect way of using the features. These measurements establish useful but variable MBON information, not exact neural information capacity.

The preflight scored the same predetermined held-out **features** later used for frozen evaluation; its thresholds and stop rule were fixed beforehand, no seed was replaced, and no outcome or reward label was used to choose a controller. The resulting performance remains conditional on that predeclared information gate.

### Feedback in the two archived studies

Training action counts by observed signature (A0/A1/A2) were:

| Archived controller | C0 | C1 | C2 | C2/A2 sampled reward versus selected-case expected reward |
| --- | --- | --- | --- | ---: |
| Original 14001 | 24/9/3 | 22/38/6 | 20/10/12 | −.257 versus −.184 |
| LinUCB 15001 | 40/0/1 | 16/45/1 | 19/6/16 | +.123 versus +.208 |

These expectations use hidden causes only after the fact. The original's A2 was sampled 21 times overall but had unfavorable C2 feedback; LinUCB tried A2 more often on C2, reducing training failures, yet still missed its final teacher gate. A zero count in a cell means no direct reward estimate for that action/signature from that controller.

## Fixed reward-learning test

Fresh seeds **16001–16002** each supplied 144 training attempts and an independent 36-case evaluation tape. The **same unchanged ridge-1, bonus-0.5 LinUCB core** learned either from three one-hot signature inputs or from the three measured MBON features. Case order, hidden causes, sampled potential outcomes, feedback rule, initialization, budget, and evaluation cases matched within seed. Only the selected action's reward updated its controller. Frozen checkpoints at **0/24/48/96/144** used copied parameters, zero confidence bonus, no learning, and the same held-out vectors. All 144 updates occurred in each arm, evaluation parameters and neural multipliers stayed unchanged, and the locked validity gate passed.

| Seed | Arm | Frozen expected reward at 0 / 24 / 48 / 96 / 144 | Final successes | Final A0/A1/A2 | Final C2→A2 |
| --- | --- | --- | ---: | --- | ---: |
| 16001 | Raw signature | .093 / .093 / .093 / .093 / **.298** | 18/36 | 20/16/0 | 0/9 |
| 16001 | MBON | .093 / .107 / .093 / .093 / **.296** | 19/36 | 16/20/0 | 0/9 |
| 16002 | Raw signature | .093 / .096 / .225 / .225 / **.225** | 22/36 | 12/24/0 | 0/11 |
| 16002 | MBON | .093 / .145 / .145 / .165 / **.173** | 23/36 | 3/33/0 | 0/11 |

**Both learning gates failed.** Each raw arm crossed +0.20 but selected A2 **zero** times at frozen evaluation, missing its required four selections. The raw-learnability prerequisite therefore failed; the fixed budget/controller did not establish full recovery learning even with the categorical signature directly available. Both MBON arms also chose A2 zero times. The 16001 MBON arm missed the 22/36 recovery and coverage gates despite +0.296 expected reward. The 16002 MBON arm missed the +0.20 expected-reward, all-action, C2-A2, and within-0.05-of-raw gates despite 23/36 realized recoveries; its expected reward trailed raw by **0.0516**. Realized successes and model expected reward diverged on these small, stochastic tapes.

Relative to the attainable information references above, the final raw learners left **.0707/.0414** expected reward unrealized on seeds 16001/16002; the MBON learners left **.0564/.0546** relative to the fixed centroid-based MBON reference. The latter is a specific analytical policy, so these differences are diagnostic gaps, not regret against an exact MBON optimum.

### Sampled feedback and coverage

Training A2 selections were sparse: **6/144 raw and 8/144 MBON** for seed 16001, **4/144 raw and 5/144 MBON** for 16002. On the key C2 cases, A2 received respectively **4, 4, 1, and 1** selected-action rewards across those four arms. The C2/A2 sampled mean rewards were **−0.14, −0.14, −0.84, and −0.84**. Yet their cause-conditioned expected means on those *selected cases*, calculated only for analysis, were **+0.126, +0.308, −0.420, and +0.308**. In particular, the 16002 MBON learner received one failed C2/A2 attempt whose expected reward was positive; neither the hidden expectation nor counterfactual rewards were available to it. The raw 16002 C2/A2 selected case had a genuinely negative hidden-cause expectation, showing that sparse sampling also changed which causes supplied feedback. This is direct evidence of limited, sometimes misleading selected-action feedback, not proof that it is the sole cause. Shared linear values and the fixed optimistic selection rule may also matter; no parameter or architecture comparison was run.

## Decision and limits

Choose **decision 1** from the protocol's interpretation options: the current MBON representation carries enough information for a useful analytical recovery policy on both new seeds, while this fixed reward budget/controller failed to learn a reliable three-action policy. The raw-observation control's same coverage failure means the study **cannot attribute the main failure specifically to MBON information loss**. It also cannot separate stochastic feedback from controller bias as a unique cause. The original and LinUCB failures remain valid and the new attempt does not rescue either. No teacher–student recovery study or demo integration is justified.

The single next research step is **stop this synthetic recovery branch before HackPrinceton** and report the failed independent-learning prerequisite. A future study would need independently grounded sensor and recovery-cost data before a new learning protocol is scientifically useful; no such study was run here.

Validation: `node --check tools/check_package_recovery_learnability.mjs` and `node tools/test_package_recovery_linucb.mjs` passed before the write-once run. To reproduce neural probes and training, copy the repository **without** the result artifact and run `node tools/check_package_recovery_learnability.mjs`; the runner refuses to overwrite the saved result. The [raw artifact](../data/package_recovery_learnability.json) contains the protocol hash, case-level neural features, training and frozen rows, controller parameters, and all gates.
