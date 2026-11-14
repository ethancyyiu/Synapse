# Synthetic package-pickup recovery: initial locked study

**. Status: stopped at the teacher prerequisite.** The [protocol](package-recovery-protocol.md) was saved before the first probe. The isolated [runner](../tools/check_package_recovery.mjs) and write-once [trial-level result](../data/package_recovery_initial.json) preserve the fixed seeds, cases, outcomes, neural features, controller scores, actions, and gates. The earlier failed [warehouse route](warehouse-navigation-report.md), RPS research, and demo were not changed.

## Operational model and what the connectome processes

This is a **synthetic one-attempt recovery decision**, not a robot trial. Four hidden pickup-failure causes produce overlapping categorical post-failure sensor signatures: alignment offset, slip/rotation, or high resistance. Regripping, stabilizing, and clearing/escalating have different cause-dependent success probabilities and explicit delay, failure, and escalation costs. The agent sees only a noisy sensor signature and its own past selected-action rewards. The case generator fixes potential outcomes for matched arms, but the policy never receives a cause, unchosen outcome, best action, or success probability. Because each signature can arise from all four causes and success is stochastic, this is distinct from deterministic RPS, though it remains a small three-cue contextual-bandit toy rather than realistic robotic recovery.

The existing real-wiring simulated VPN→KC→MBON pathway processes the three categorical signatures. The three pooled MBON rates feed the unchanged external hybrid contextual-bandit controller. KC→MBON multipliers stay frozen; its own reward changes only controller values. A future teaching stage would encode teacher preference into stimulated DNge175 spikes and decode actual counts, but this stage stopped before any DN probe or student training. The detailed downstream DN connectivity was not tested.

## Locked prerequisites and results

| Gate | Locked requirement | Initial result |
| --- | --- | --- |
| Task observability | Three distinct expected-best cue actions, each advantage ≥0.03; balanced 144/36 independent case streams | Passed. Cue 0/1/2 best actions 0/1/2; expected rewards by action were `(0.350,0.110,-0.460)`, `(0.072,0.358,-0.441)`, `(-0.136,-0.104,0.044)`. |
| Neural sensory | VPN 6/6 per cue; held-out MBON centroid ≥14/18 and ≥4/6 per cue | Passed: VPN 6/6 each; MBON **17/18**, by cue **6/6, 6/6, 5/6**. Analytical decoder fitted on separate calibration probes only and never installed. |
| Teacher learning | Frozen expected net reward ≥0.20 and ≥0.10 above initial; ≥22/36 realized recoveries; ≥4 choices of each action; controller change; neural weights unchanged | **Failed.** Expected reward **0.093→0.1943** (gain +0.1013), recoveries **22/36**, final action counts **20/16/0**. Action 2 received **21** training choices but zero frozen choices. Controller changed; KC→MBON multipliers stayed one. |

The 0.1943 expected-reward result missed the fixed 0.20 floor by 0.0057, and action 2 missed its four-choice floor. In the frozen 36 cases, the cue-2 group had **6/15** recoveries; the teacher chose action 0 on all 15 cue-2 cases, although action 2 has the highest *design-time cue-conditioned expected reward*. This identifies the **observed policy limitation**. It does not establish why the teacher's training produced that preference or that a longer budget would fix it. Training reward is not a frozen learning metric: 54/144 training attempts succeeded, versus 22/36 in the separate frozen held-out evaluation. Held-out expected reward is an analysis metric calculated from the synthetic cause-specific outcome table, not information available to the fly.

The source-preference gate, DN receiving check, independent Fly B learning, five-arm teacher/control comparison, and independent confirmation were **not run**. Therefore no teacher-specific improvement, retained student policy, reduced student failures, or practical robot utility is established. The prerequisite failure is final for this locked protocol; no threshold, seed, action cost, or learning setting was adjusted.

## Reproduction and integrity

The first deterministic invocation completed through the teacher gate but the sandbox blocked creation of the result file because the renamed repository lay outside its configured writable root. The identical locked invocation was then permitted solely to save the write-once artifact. No experimental setting changed between invocations. `node --check tools/check_package_recovery.mjs` passed. For a full reproduction, use a **copy** without the existing result file, then run `node tools/check_package_recovery.mjs initial`; the runner refuses to overwrite archived results. The stored protocol SHA-256 ties the output to the locked protocol. No confirmation command is warranted because the initial teacher gate failed.

**Conclusion:** the sensory abstraction was distinguishable by the simulated fly pathway and the teacher showed some reward-driven improvement, but it did not meet the predefined three-action teacher prerequisite. This package-recovery application should not be added to the HackPrinceton demo as a demonstrated teaching or robot-recovery success.

## Read-only diagnosis of the failed teacher gate

This section analyzes the **unchanged** initial artifact. It is post hoc and does not amend the prospective protocol or turn the failed gate into a pass. Reproduce from the saved JSON and `tools/check_package_recovery.mjs`; no neural simulation or retraining is required. The controller sees a noisy cue and simulated MBON features, never the hidden cause or unchosen outcome.

### Was clearing optimal from the available observation?

Yes. With the protocol's equal hidden-cause prior, Bayes weighting the documented cue probabilities and net-reward formula gives:

| Observable cue | Expected reward: regrip A0 | Stabilize A1 | Clear/escalate A2 | Best expected action |
| --- | ---: | ---: | ---: | --- |
| C0 alignment offset | **+0.350** | +0.110 | −0.460 | A0 |
| C1 slip/rotation | +0.072 | **+0.358** | −0.441 | A1 |
| C2 high resistance | −0.136 | −0.104 | **+0.044** | A2 |

The C2 posterior probability of an actual obstruction is 2/3, but the high delay and escalation cost make A2's *absolute* expected reward small. Its expected advantage over the next-best action is about **0.148** under the design distribution. Thus A2 is neither universally good nor genuinely suboptimal: it is best only for C2. On this finite held-out set, its cause-conditioned expected reward on C2 was **−0.017**, still above A1 **−0.067** and A0 **−0.087**. The finite realized potential outcomes could rank actions differently; those outcomes were not visible before action selection.

### Training exposure and feedback

Each cell below is `A0/A1/A2` action counts on the **observed** cue. Phases are consecutive groups of 36 attempts, not frozen checkpoints.

| Training attempts | C0 | C1 | C2 | Exploratory selections, all cues |
| --- | --- | --- | --- | ---: |
| 1–36 | 9/1/0 | 8/8/1 | 8/0/1 | 4 |
| 37–72 | 4/4/0 | 2/16/0 | 3/0/7 | 6 |
| 73–108 | 9/2/1 | 7/2/1 | 5/6/3 | 6 |
| 109–144 | 2/2/2 | 5/12/4 | 4/4/1 | 8 |
| **Total** | **24/9/3** | **22/38/6** | **20/10/12** | **24/144** |

The fixed epsilon-selection RNG can be reconstructed from the saved keys; it reproduces **144/144** actions, with 24 exploratory selections. A2 was selected **21** times overall, seven through exploration and fourteen as the greedy choice. On C2, A2 was chosen **12** times, three exploratory and nine greedy. It was not absent from training. It succeeded **6/21** overall and **5/12** on C2, producing six positive and fifteen negative selected-action prediction updates overall. Mean observed reward for C2/A2 was **−0.257**, close to C2/A0's **−0.250**, despite the design-time expectation favoring A2.

The selected C2/A2 cases were unusually unhelpful for learning that action: only **5/12** were actual obstructions, versus **28/42** among all C2 training cases; C2/A0 saw **18/20** obstructions. This is an imbalance in the finite *selected feedback*, not evidence that the fly knew the hidden cause. Across **all** 42 C2 training cases, A2's unobserved potential-outcome reward averaged **+0.027**, versus A0's **−0.027**; the learner could not see this counterfactual. It received sparse, noisy, action-dependent feedback for the only cue where A2 was useful. The late 36 attempts included just one C2/A2 selection, which failed.

### Learned values, representation, and the missed threshold

On frozen held-out C2 probes, mean Q scores were **A0 +0.156, A1 −0.262, A2 −0.242**. The mean A2-versus-best margin was **−0.398**, and A0 won **15/15** C2 decisions. A post hoc nearest-centroid check fitted on the teacher's saved training MBON features recognized **13/15** frozen C2 cues and **31/36** cues overall. This check was not the teacher's policy; it shows the C2 neural features were usually distinguishable, while leaving some sensory overlap possible. The archived student sensory gate (17/18) used a **different seed** and alone cannot establish the teacher's recognition quality.

Replaying the exact selected-action update `step = 0.03 × (reward − Q[action])`, `b[action] += step`, `W[action,j] += step × feature[j]` from the saved rows reproduces every archived pre-update score, update step, and final controller parameter with zero recorded numerical discrepancy. The negative A2 Q and its nearly flat score across cues are therefore consistent with the recorded rewards and the shared linear readout; no update implementation error is established. Negative A2 feedback on C0/C1 and the unrepresentative C2/A2 cases plausibly contributed. The records do not isolate the relative causal contributions of stochastic outcomes, training order, shared-feature generalization, and remaining neural variability.

The initial frozen expected net reward was **0.093** and final was **0.1943**. The observable-cue expected-best policy would score **0.2418** on the same held-out cases. Replacing only Fly A's C2 choices with A2 *analytically* raises its expected reward to **0.2234**, above the locked **0.20** floor; no action was actually replaced in the experiment. Its C0 choices also left some expected reward unrealized. Thus the missed expected-reward gate reflects a deficient *learned policy relative to this task's expected objective*, not an environment in which clearing is never worthwhile. The cost/noise assumptions do make the C2 learning signal weak and the 36-case evaluation variable; no unique algorithmic cause is proven.

The original all-action criterion was meaningful for the **specified** reward model: each action is best for an observable cue, and C2 appeared in 15/36 held-out cases. It is a demanding task-specific coverage gate, not a universal requirement for recovery policies. It remains failed as written. **Recommendation: stop this package-recovery research branch under the current synthetic protocol.** Without independently grounded robot failure frequencies and action costs, adjusting the synthetic economics or retraining after this negative result would mainly tune a toy task toward success. Preserve this result and the established RPS demonstration.
