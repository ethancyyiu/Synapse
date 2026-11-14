# Localized RPS circuit disruption: neural effect without behavioral impairment

**. The [prospectively locked feasibility test](neural-disruption-feasibility-protocol.md) failed its behavioral gate.** A fixed 50% reduction of KC→MBON synaptic transmission into one existing action-related MBON group changed measured neural activity in two independently trained Strategy B students, but changed **0/69 observed-cue frozen actions and 0/72 wins** in each. No learning, teacher message, or recovery experiment followed. The validated Strategy A/B teaching and failed package-recovery artifacts remain unchanged.

## Architecture audit: where the connectome enters

```text
Previous opponent move → 150 Hz VPN cue → simulated KC/MBON spikes → three pooled MBON rates
                            │                                  │
                            └→ measured VPN counts → external cue-head selection
                                                               ↓
                                         external reward-trained Q scores → frozen greedy action

Trained Fly A: MBON-driven Q preference → engineered DN3141/3165 drive in Fly B
           → actual DN spike counts → software decoder → external imitation update
Fly B: its own action + RPS reward → external selected-action Q update
Frozen evaluation: exploration, DN teaching, and all learning disabled
```

The real-wiring simulator propagates VPN activity through KC and MBON pathways and generates the measured spikes. Both Fly A's contextual-bandit Q and Fly B's cue-specific Q heads are **external hybrid controllers**. Fly B's head is routed by VPN counts, while its three MBON-rate features enter the action values within that head. KC→MBON plasticity is frozen in the validated RPS teaching system. The DN message is directly stimulated, retained by a measured-spike buffer, decoded in software, and used in an external imitation equation. The prior [neural-versus-digital ablation](connectome-contribution-ablation-report.md) found identical learning from measured spikes and direct delivery, so it established spike transport rather than a need for downstream DN wiring. A neural-rate change can alter Q values without crossing a decision boundary; an action change must be measured separately.

## One selected disruption and controls

Before new lesion outcomes, the protocol selected the **9,210 real-wiring KC→MBON edges ending on 14 group-0 MBONs** (`net.plastic[0]`) out of 28,238 KC→MBON edges. Their existing simulator multipliers were set to **0.5** only in the disrupted arm. Intact used all multipliers one; sham explicitly set the same target to one. This partial, localized synaptic attenuation preserves input and some downstream activity. The simulator's action-group label is engineered; it is not proof of a biological compartment or a model of real injury.

Saved trained Strategy B student controllers **11015/11016** were reconstructed from their archived final weights, with their original frozen opponent seeds and keyed random streams. Each arm ran the original 680 ms schedule, 150 Hz VPN sensory cue, 50 ms count reset, 240 ms decision, and 620 ms disabled update. No DN message was sent. Intact exactly reproduced all archived 72 actions, three MBON rates, three standardized features, three scores, and **70/72 wins** for each student. Sham matched intact trial for trial. Controller weights and all untargeted synaptic multipliers stayed unchanged.

## Paired frozen results

Means below use the 69 rounds per student with an observed previous opponent move. Each held-out cue occurred 23 times. All observed-cue actions were correct before and after disruption.

| Student | Group-0 MBON rate, sham → disrupted | Reduction | Active KCs, sham → disrupted | VPN cue routes | Action flips | Frozen wins, sham → disrupted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11015 | .480 → .360 | **25.0%** | 7.29 → 7.23 | 69/69 | **0/69** | **70/72 → 70/72** |
| 11016 | .470 → .364 | **22.5%** | 7.77 → 7.41 | 69/69 | **0/69** | **70/72 → 70/72** |

Other pooled MBON rates also shifted modestly and are saved per trial; this is not perfect circuit isolation at the output level. Mean absolute Q-score changes by action were **.171/.200/.178** for 11015 and **.184/.190/.153** for 11016. Nevertheless, the mean correct-action margin remained near **2.00** and **1.94**, respectively, with no greedy-boundary crossing. For both students, cue-specific wins stayed **23/23, 23/23, 23/23**; the three cue-less first rounds account for the remaining difference from 72. The observed result is a reproducible neural and score perturbation with **no behavioral impairment**.

## Locked gate ledger and decision

| Predeclared requirement, both students | Outcome |
| --- | --- |
| Intact archived replay and exact sham; fixed controller and target-only multiplier change | **Pass** |
| Group-0 rate reduction ≥20%, with remaining rate >.05 | **Pass:** 25.0%, 22.5%; remaining .360/.364 |
| VPN routing 69/69 and active KC count ≥50% of sham | **Pass:** 69/69 each; KC ratios .992/.953 |
| ≥4/69 action changes **and** ≥4/72 fewer frozen wins | **Fail:** 0 action changes, 0 win loss in both |

The joint feasibility gate **failed**. The architecture permits neural activity to influence external Q scores, and this intervention did so. It did **not** establish a damaged behavioral policy or a recovery target in these trained students. Because their cue routing and all observed-cue choices remained correct, teacher-assisted recovery would have no impairment to repair under this lesion. No stronger lesion, different cell group, new seed, relearning, or teacher-control study was run. A total spike-count reduction alone is insufficient to justify recovery. The single next step is **stop this disruption route and retain the negative feasibility result**. A future injury-and-recovery program would need a separately justified circuit target and prospective behavioral gate, not a post hoc adjustment of this one.

## Artifact and checks

The [write-once trial artifact](../data/neural_disruption_feasibility.json) includes all 432 intact/sham/disrupted frozen rows across the two students, actual MBON counts and VPN routing, paired actions and rewards, Q scores, target multipliers, metrics, and gates. Its protocol SHA-256 is `6f23c98742704fd214c31d41cc54c4913986f64074eb944294e971dacc956121`. Focused syntax and preflight checks passed. A saved-row audit verified the hash, 72 rows per arm, exact intact/sham matches, zero action flips, and cue-specific outcomes.

From the repository root: `node --check tools/check_neural_disruption_feasibility.mjs`; `node tools/check_neural_disruption_feasibility.mjs preflight`. Reproduce the frozen simulation only in a copy without the write-once artifact using `node tools/check_neural_disruption_feasibility.mjs`. No Git operation, model-rule change, demo edit, or training occurred.
