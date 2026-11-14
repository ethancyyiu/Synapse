# Uncertainty-aware synthetic package recovery: locked initial comparison

**: valid initial comparison, failed teacher-learning gate.** The [new protocol](package-recovery-uncertainty-protocol.md) was saved before any new training. The [isolated LinUCB readout](../tools/package_recovery_linucb.mjs), [runner](../tools/check_package_recovery_uncertainty.mjs), and [write-once trial-level result](../data/package_recovery_uncertainty_initial.json) preserve the configuration and measurements. The original package-recovery failure, successful RPS architecture, and HackPrinceton demo were not modified. This is a synthetic one-attempt decision task, not measured robot performance.

## Controller and implementation validity

The original controller initializes three linear MBON-reward predictors to zero, selects with fixed epsilon 0.20, and updates only the selected action by `0.03 × (reward − predicted reward) × [features,bias]`. A2 received 21/144 choices but unfavorable feedback in the prior study. The new **ridge LinUCB** receives the *same three standardized simulated MBON rates* plus a bias. For each recovery action it tracks `A=I+Σxxᵀ`, `b=Σrx`, predicts `xᵀA⁻¹b`, and adds a fixed `0.5 sqrt(xᵀA⁻¹x)` uncertainty bonus **only during training**. It updates only the action actually attempted, from that attempt's own net reward. No hidden cause, optimal action, or unchosen outcome enters its readout. Its confidence proxy captures sparse feature/action coverage, not calibrated outcome variance. Frozen evaluation chooses the highest posterior mean without the bonus or any learning. KC→MBON synaptic multipliers remain one; the readout is an additional engineered hybrid learning component.

Pre-run checks passed for action selection, decreasing uncertainty after exposure, selected-action updates, the MBON transform, deterministic replay, and unchanged frozen parameters. The first unit-test assertion incorrectly expected UCB to leave an action after a positive reward; it was corrected **before the neural run** to test a negative reward, with no controller or protocol change. Saved-run validity passed: **144/144** matched case IDs and neural input vectors, common held-out cases, zero initial means, 144 updates per arm, unchanged frozen controller/neural weights, and mean-only frozen LinUCB decisions. The protocol hash matches the result artifact.

## Frozen independent held-out learning curves

Both arms trained on the same 144 cases with neural seed **15001**, then were evaluated on the same **36** independently seeded cases at every checkpoint. Values below are frozen cause-conditioned **expected net reward per case**; successes are realized outcomes on that held-out tape. Training success is not substituted for frozen performance.

| Training attempts | Original expected / successes | LinUCB expected / successes | LinUCB C2 choices of A2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.093 / 14 | 0.093 / 14 | 0/12 |
| 24 | 0.200 / 18 | 0.126 / 16 | 2/12 |
| 48 | 0.093 / 14 | 0.250 / 21 | 3/12 |
| 96 | 0.118 / 12 | 0.252 / 20 | 7/12 |
| 144 | 0.192 / 18 | **0.211 / 19** | **7/12** |

The predeclared early mean across attempts 24/48/96 was **0.2096** for LinUCB versus **0.1370** for the original, an advantage of **+0.0726** (speed threshold +0.04). This curve is uneven: the original was better at attempt 24, and LinUCB declined from 0.252 at 96 to 0.211 at 144. Final frozen action counts were original **16/19/1** and LinUCB **16/12/8** for A0/A1/A2. On C2, original chose A2 **1/12** and recovered **5/12**; LinUCB chose A2 **7/12** and recovered **6/12**. These are individual-seed results, not reliability evidence.

## Costly training attempts and locked gates

| Training measure, 144 matched attempts | Original | LinUCB |
| --- | ---: | ---: |
| Failed recovery attempts | 82 | **68** |
| Mean realized net reward, including delay/escalation cost | 0.0876 | **0.2101** |
| Total delay seconds | 466 | 498 |
| Escalations | 13 | 18 |
| Training action counts A0/A1/A2 | 81/50/13 | 75/51/18 |

LinUCB had **14 fewer failures** and **+0.1225** mean net reward despite 32 extra seconds of cumulative delay and five extra escalations. Its confidence bonus induced broader A2 coverage; the original epsilon sampler made 27 exploratory selections, while LinUCB made deterministic optimistic selections. The locked learning-speed/cost gate passed.

The **joint teacher-learning gate failed**. LinUCB passed its absolute final expected reward floor (**0.211 ≥0.20**), own-baseline gain (**+0.118 ≥0.10**), and all-action coverage (**16/12/8 ≥4 each**). It missed three required conditions: final advantage over the matched original was **+0.0198**, below **+0.05**; held-out recoveries were **19/36**, below **22/36**; and C2 A2 selection was **7/12 =58.3%**, below **60%** (which requires at least 8/12). These thresholds were not changed. The improvement in training attempts did not translate into a large enough or reliable enough final frozen policy advantage.

## Decision and limits

The initial experiment is a **valid negative teacher-gate result with a positive learning-speed signal**. It cannot establish that LinUCB is generally better, that uncertainty is the uniquely causal ingredient, or that a teacher can teach a student through DN spikes. The task remains a synthetic categorical sensor abstraction with stipulated costs and success probabilities. Because the initial joint gate failed, the locked independent seed **15002** was **not run**; no DN receiving or five-arm student experiment was run. The single next step is to **stop this branch under the current locked method** and retain the result for later independent consideration, rather than search new bonuses or architectures before HackPrinceton.

Focused validation: `node tools/test_package_recovery_linucb.mjs` and `node --check tools/check_package_recovery_uncertainty.mjs` passed. The runner is write-once. To reproduce the simulation, use a copy without the result file and run `node tools/check_package_recovery_uncertainty.mjs initial`; do not overwrite the archived artifact.
