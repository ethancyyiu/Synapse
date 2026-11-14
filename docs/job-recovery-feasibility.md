# Data-center job recovery: feasibility assessment

**Decision: NO-GO for a new Synapse research or demo implementation before HackPrinceton.** Job recovery is a credible operational decision, but the current repository has neither grounded job-failure data nor a validated neural encoding for its required multi-field context. The previous synthetic package-recovery task failed independent three-action learning even when its categorical observation was supplied directly. No new simulator, neural probe, training run, or demo change was made for this assessment.

## Operational problem and evidence

After a failed batch job, choose one bounded response: **retry on the same worker**, **move to an available different worker**, or **escalate for investigation**. The choice trades off time to completion, repeated resource consumption, migration overhead, and human attention. The candidate is limited to jobs that can be safely retried or moved and for which a different worker is available; production scheduling, idempotency, security, and hard safety policies remain outside the learning agent.

This is a real class of operational problem. [Kubernetes Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/) use backoff limits and pod-failure rules; their documentation distinguishes failures that should fail a job from disruptions that can be ignored for retry counting. [AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/job_retries.html) supports retry decisions based on exit codes and failure reasons. [Google Cloud Batch](https://docs.cloud.google.com/batch/docs/automate-task-retries) names preemption, host errors, and transient network errors as cases where retry can help. [Google SRE guidance](https://sre.google/sre-book/addressing-cascading-failures/) describes how aggressive retries can amplify overload. These sources establish the *problem and existing mechanisms*, not that Synapse would outperform them. No prevalence, cost saving, or novelty statistic is claimed.

### What the agent must observe

The smallest defensible pre-decision record includes:

| Field | Why it matters | Boundary |
| --- | --- | --- |
| Failure category and exit/status reason | Separates likely transient infrastructure faults from persistent application/configuration faults | No category is treated as a correct-action label; categories remain noisy and overlapping |
| Attempt count and last retry result | Prevents an unbounded same-worker retry loop | Past outcome is observable history summarized in the current context |
| Current worker resource pressure and alternate-worker availability | Distinguishes local exhaustion from job-wide demand and tests whether migration is feasible | A benchmark must restrict cases to available, eligible alternatives or define explicit action masks |
| Job resource request and priority/deadline | Makes delay and wasted compute costs meaningful | Priority changes utility, not hidden failure cause |

The action is chosen **before** the next attempt outcome. A failure code alone is insufficient: a resource error could reflect a saturated worker, an oversized job, or bad configuration. The agent receives only the chosen action's later completion/failure, latency, resource cost, and escalation outcome. It does not observe what the two unchosen actions would have done. This is a contextual decision problem because actions have different costs and stochastic consequences for the *same* observation; the objective is expected operational utility, not matching a supplied three-class answer. A short fixed outcome horizon is necessary. If escalation cannot be assessed within it, the task requires delayed-credit machinery beyond the existing contextual bandit.

## Minimal benchmark design, if reconsidered later

One synthetic decision follows the first failed job attempt. Latent causes could include transient worker/network fault, worker-local resource exhaustion, persistent job configuration error, and job-wide resource mismatch. A **noisy joint telemetry record** would expose the fields above, with several distinct records per cause and overlap across causes. All three actions must be feasible on every included case. Generate fixed training jobs and held-out job families and worker pools; split by job/worker identity rather than randomizing repeated attempts from the same job across sets. Prespecify a finite outcome horizon and a cost ledger:

\[
U = V_{\mathrm{completed\ by\ deadline}}
  - C_{\mathrm{compute}}
  - C_{\mathrm{queue/migration\ delay}}
  - C_{\mathrm{repeat\ failure}}
  - C_{\mathrm{human\ escalation}}.
\]

Report completed jobs, repeated failures, time to completion, worker-seconds, migrations, escalations, and mean net utility separately. Costs and outcome probabilities must come from an independently documented trace or a prospectively declared sensitivity range, not be adjusted until a learner passes. A three-profile lookup table with one nearly deterministic best action per profile would merely repackage RPS or the failed package-recovery toy. The useful benchmark needs observable overlap, action-dependent costs, and worker/retry context, while keeping the expected-best action separated enough from its competitors for finite reward feedback to be informative.

Before any teacher study, enumerate the stipulated model to check that **each action is best in at least one observable context**, the best-versus-second-best expected-utility gaps exceed the uncertainty of the planned sample, and each context/action receives enough selected feedback. Then run a raw-observation learner, a simple rule policy, and a conventional contextual bandit on matched cases. The old package-recovery result is the warning: its clearing action received only **1–4 relevant selected rewards per arm** in the latest fixed-budget test, and neither raw nor MBON learners used it in frozen evaluation ([report](package-recovery-learnability-report.md)). No job-recovery training budget or numerical success gate is locked here because neither a data source nor cost scale has been established.

## Fit to the existing neural teaching system

| Layer | What is currently supported | Missing for job recovery |
| --- | --- | --- |
| Original observations | Three discrete RPS cues or an engineered categorical recovery signature can drive existing VPN lanes | No validated encoding of failure reason **plus** retry history, load, alternative capacity, and priority |
| VPN→KC→MBON simulation | Real simulated spikes and three pooled MBON rates respond to selected cues | No evidence these three rates preserve the required joint telemetry; the prior two-sensor graded encoding recognized only **18/36** held-out classes ([sensor report](sensor-classification-report.md)) |
| Engineered controller | Reward-trained contextual heads use selected-action feedback; KC→MBON weights remain frozen | Current cue-specific heads route among three contexts; a richer state may need a separately validated representation, and delayed escalation may not fit a one-step update |
| Teacher message | Learned Fly A action preference can be encoded in measured DN3141/3165 spikes and decoded for a later student imitation update | No job-recovery teacher policy or message-quality evidence exists. Advice would affect *later learning*, not the already chosen recovery action |

The [Strategy A audit](action-preference-independent-audit.md) and [Strategy B audit](action-preference-strategy-b-independent-audit.md) support faster early **disconnected** RPS learning against four controls on two fixed rules, using an engineered reward-plus-imitation student. They do not establish job-recovery transfer. A matched [connectome ablation](connectome-contribution-ablation-report.md) found measured DN-spike and direct digital delivery of the same teacher preference produced identical student behavior and frozen curves. Thus a future job benchmark could test whether useful *learned information* crosses a constrained neural channel; it should not presume a benefit from the channel or detailed DN connectivity itself.

The smallest sensory test before implementation would freeze the existing simulator and specify **one** causal, label-blind encoding of two pre-decision telemetry fields (for example failure category and worker pressure) into existing VPN inputs. Fit any analytical recognition rule on calibration recordings only; compare raw-field and three-MBON held-out discrimination across the full crossed conditions, with independent neural seeds, no rewards, and unchanged weights. A failed information gate stops before teacher or student learning. The prior graded two-VPN sensor test failed at **18/36**, so success cannot be assumed or obtained by trying encodings against the held-out set.

## Comparison and teaching potential

| Approach | What it can establish or provide | Limitation for this project |
| --- | --- | --- |
| Fixed retry/backoff | Simple bounded recovery for transient failures | Can waste attempts on persistent faults; a strong operational baseline |
| Rule-based exit-code policy | Uses known failure semantics and operator constraints, as production systems already do | May miss interactions between noisy telemetry, worker state, and costs |
| Conventional contextual bandit | Learns expected utility from each chosen action's feedback | Needs exploration/coverage and a validated observable context; likely the strongest non-neural learning baseline |
| Direct digital transfer | Passes the same teacher preference without spike simulation | Prior RPS ablation matched neural delivery exactly; necessary control for any neural claim |
| Synapse hybrid neural teaching | Tests whether independently reward-trained agents can share a learned preference through measured spikes and retain a policy after disconnection | Requires independent job learning, usable neural features, good teacher advice, and a controlled five-arm speed comparison; none has been shown on this task |

If those prerequisites ever pass, compare trained teacher, no message, untrained teacher, globally shuffled trained messages, and generic messages, plus direct digital delivery of the same preference. Match student starts, telemetry, rewards, action opportunities, and message exposure; verify the shuffled/generic tapes do not accidentally carry strong action information. Evaluate frozen, disconnected policies at predeclared early checkpoints and at the end. The primary endpoint should be **early retained net utility and failed attempts**, with completion, delay, and escalation reported separately. A teacher that simply encodes the simulator's hidden optimal action would be an answer oracle, not learned teaching. An independently learning student that reaches a ceiling almost immediately would leave no measurable speed headroom.

## Decision and single next step

**NO-GO for the current HackPrinceton scope.** Real-world relevance is strong, but independent learnability from the proposed telemetry is untested; the prior synthetic recovery learner failed even with raw categorical cues. Multi-field VPN/MBON representation is unvalidated, short-horizon escalation feedback may be incomplete, and the three-cue implementation would risk becoming another action-labeled toy. Adapting the existing architecture faithfully would require more than a small demo integration. None of this weakens the archived RPS findings.

**One next step, after the event if pursued:** obtain a small anonymized failed-job log and perform a read-only audit of **pre-decision telemetry, action coverage, and observed recovery outcomes**. If that log cannot support causal context and outcome definitions for all three feasible actions, stop rather than inventing favorable synthetic probabilities. This assessment does not authorize or begin a new simulator or neural study.
