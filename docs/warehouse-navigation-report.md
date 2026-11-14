# Warehouse-route proof of concept: stopped at teacher feasibility

** locked initial result:** The toy warehouse environment and the predefined frozen sensory test passed. The reward-trained teacher then missed its **9/12 successful-delivery** gate, scoring **8/12**, so the study stopped before neural-message transmission, Fly B learning, five-arm teaching controls, or independent confirmation. The failed gate remains failed. No earlier RPS result or HackPrinceton demo was changed.

## Task and information

The [prospective protocol](warehouse-navigation-protocol.md) defines a four-tile route in a 5×5 warehouse: start **S → J → K → delivery bay G**. All other tiles are shelves, and a wall blocks the direct S→G shortcut. Actions **0/1/2** mean east/north/west. The robot gets +0.4 for advancing, +1 for delivery, −0.4 for backtracking, −0.8 for collision, and −0.3 after a four-decision episode with no delivery. After a delivery, another package starts at the same predetermined starting cell so all arms would have the same four decision opportunities per episode. Training uses two translated layouts and 36 balanced-start episodes; frozen evaluation uses a third translated layout and 12 balanced-start episodes. Translations preserve the same route topology; they do not test new obstacle configurations.

Only a visible location marker enters the policy: S→VPN cue 2, J→cue 0, K→cue 1, each stimulated at 150 Hz. This is an observable position marker, not the optimal action; no goal direction, transition table, answer, or layout ID is passed to the fly. A memoryless policy can solve the fixed route in at most 3/2/1 moves from S/J/K within the four-decision horizon. The environment's transition enumeration and balanced schedules passed before neural training.

## Frozen sensory and teacher gates

The student-seed frozen sensory probes used the existing `50–240 ms` spike-count window, three pooled MBON rates, and actual VPN-count cue routing. Six calibration and six disjoint held-out probes per cue were fixed before execution. A calibration-only nearest-centroid analysis recognized **16/18** held-out MBON vectors (cue 0/1/2: **5/6, 6/6, 5/6**) against the locked **≥14/18, ≥4/6 per cue** gate. The actual VPN router identified **6/6 for each cue**; centroids were distinct, features finite, and KC→MBON multipliers unchanged. This analytical decoder was never installed in the teacher or student.

Fly A then received **144** navigation decisions across 36 fixed episodes, using the existing three-MBON contextual bandit with `alpha=.03`, `epsilon=.20`, and frozen neural synapses. Its controller weights changed from its own scalar rewards. The locked teacher gate required at least **9/12** successful held-out episodes, **2/4 from each start**, four selections of every action, and weight integrity. It failed:

| Frozen start | Successful episodes | Required |
| --- | ---: | ---: |
| S, longest route | **1/4** | ≥2/4 |
| J | 3/4 | ≥2/4 |
| K | 4/4 | ≥2/4 |
| **Total** | **8/12** | **≥9/12** |

The frozen teacher selected east/north/west **14/5/29** times over 48 decisions, so the action-frequency subgate passed. At J it chose east/north/west **3/5/9** times over 17 visits; west backtracks. During training it chose the rewarding north move at J **12/48** visits, so absence of any rewarded north opportunity is not established. Training episode successes were **6/12, 7/12, 6/12** across successive thirds, with no clear late upward trend. These measurements localize the observed policy errors to the J decision, especially for episodes starting at S, but do not prove whether trial-to-trial MBON variability, learned value boundaries, or the task's reward structure caused them. No initial frozen teacher checkpoint was saved, so changed weights and training successes alone do not quantify independent improvement over an untrained teacher.

## Stopping, interpretation, and reproduction

The source-preference table, DN receiver checks, no-message student, five-arm comparison, and confirmation were **not run**. Thus this study establishes a memoryless, reproducible toy route and distinguishable simulated sensory cues, **not** successful teacher navigation, neural communication on this task, student learning, retained teaching, or warehouse-robot utility. Existing RPS studies demonstrate DN transmission separately; they cannot be attached to this untested warehouse transfer. This route is close to a three-state contextual task and lacks realistic perception, changing obstacles, and physics.

The write-once [initial artifact](../data/warehouse_navigation_initial.json) contains the protocol hash, all environment transitions and schedules, 36 sensory probes, **144 teacher training rows**, and **48 frozen teacher rows**. Its SHA-256 is `CFD278E1A37CB248D8E6B98983EF10D4AFF57D76C92F752C77AB3BDF803D6C57`; embedded protocol SHA-256 `EA6CD05313BD76CB5F399BE5E8C9D39D55AFF03C8D720CCCB7C9D275EE3C732F` matched the locked document. The runner refuses to overwrite an existing result. Reproduce **only in a copy lacking the artifact** with `node --check tools/check_warehouse_navigation.mjs` and `node tools/check_warehouse_navigation.mjs initial`. Confirmation is blocked by the initial status. No parameter, seed, action code, connectome, neural dynamic, dopamine rule, RPS runner, archived artifact, or demo was changed.

**Next step:** close this locked warehouse attempt at the teacher gate and prepare the validated RPS demonstration for Princeton. A later warehouse study would need a distinct, prospective rationale for the J-decision failure; extending training or retuning this failed protocol to seek a pass is not supported by the saved curve.
