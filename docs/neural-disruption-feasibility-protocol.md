# Prospective RPS KC→MBON disruption feasibility

**Locked  before frozen lesion simulations.** This is one new feasibility experiment. The archived Strategy A/B teaching and package-recovery results remain unchanged. No controller training, teacher message, or recovery experiment is part of this test.

## Causal rationale and selected circuit

The validated hybrid RPS student senses the public previous opponent move through VPN spikes, routes it to one of three external cue heads by measured VPN counts, and uses three pooled MBON spike rates as action-value features. Its trained cue-head parameters were learned from its own RPS rewards and, in the trained arm, decoded DN advice. Frozen decisions use the MBON features with exploration and learning off. Thus the connectome's VPN→KC→MBON activity can affect frozen action scores; the external Q and imitation equations are outside the connectome. The DN message is counted from directly stimulated simulated DNge175 3141/3165, retained in a software buffer, and decoded externally. DN downstream synapses are not required by that input path; disrupting them would not isolate this question.

Select **only the KC→MBON synapses ending on the 14 MBONs in simulator action group 0**, identified by `net.plastic[0]` under real wiring. The subgraph has **9,210** such edges of 28,238 KC→MBON edges. This fixed group is chosen by its existing readout role and connectivity, before any new lesion outcome; no cell or group search is allowed. It is an engineered action grouping, not a claim that this entire group is a single anatomical compartment. The partial lesion sets the existing multiplicative synaptic factor `net.m[k]` to **0.5** for these 9,210 edges, leaving every other edge at one. It does not alter neurons, wiring, neural equations, or stored weights. This mimics reduced KC input to one MBON readout group while preserving half of that input.

## Locked frozen comparison

Use the saved **trained Strategy B** student heads from the independent initial and confirmation artifacts: student **11015** and student **11016**. Do not retrain or select among students. Reconstruct each saved `finalWeights` exactly. Use that student's archived three 24-round opponent evaluation seeds (11015: **91000/91001/91003**; 11016: **91004/91006/91009**) and original trial-keyed neural/controller random streams. For each student, run three separate frozen network/controller instances:

1. **Intact:** all synaptic multipliers one.
2. **Sham:** explicitly set the same target edge multipliers to one.
3. **Disrupted:** set only those target multipliers to 0.5.

Each condition has the same 72-round Strategy B sequence and original 680 ms round schedule. At every round start, reset that student's neural state and key streams as in the archived frozen runner. Drive the observed previous-move VPN lane at 150 Hz from 0–240 ms; reset ordinary counts at 50 ms; at 240 ms, record lane counts, active KC count, 42 MBON counts and three pooled rates, standardized MBON features, cue-head route, three external Q scores, winning-action margin, and greedy action. At 280 ms, calculate the ordinary RPS result for analysis. At 620 ms, call no learning; drive no DN target at any time. Preserve the full schedule to 680 ms. Record all 72 trial rows, per-cue wins and action changes paired by student/episode/round. The correct RPS action is used **only to score** decisions, never supplied to a controller.

First require intact and sham actions, MBON features, and wins to reproduce the saved frozen checkpoint at round 288 exactly; if reconstruction fails, stop without interpreting a lesion. The sham must be exactly identical to intact on matched rows. Verify the only changed simulator parameters are the 9,210 selected `m[k]` values, student controller parameters remain byte-identical, no DN spikes are deliberately injected, and VPN cue routing is correct on all 69 observed-cue trials per student.

## Primary gate and decision

The partial lesion is a **meaningful, reproducible impairment** only if **both** archived students meet all of the following, without changing the rule after results:

- Their mean group-0 pooled MBON rate across the 69 observed-cue decision windows is at least **20% lower** than matched sham and remains **above 0.05 spikes per MBON**; the other two pooled groups and KC recruitment are reported, not used to select a favorable target.
- Intended VPN cue routing stays correct on **69/69** observed-cue trials, and mean active-KC count stays at least **50% of sham**. This preserves an interpretable sensory representation.
- At least **4/69** observed-cue greedy actions change relative to matched sham, and frozen RPS wins fall by at least **4/72**. A neural-rate change without behavioral impairment fails the recovery-readiness gate.
- Intact/sham exact replay, fixed weights, and identical opponent/random streams pass their integrity checks.

Report every measurement even if one gate fails. A pass would define a damaged policy and intact target against which a **separately locked** recovery study could later compare independent relearning, trained teacher assistance, untrained teacher assistance, shuffled advice, and direct digital assistance, under matched outcomes and frozen disconnected evaluation. A pass does **not** establish recovery or biological injury. If any gate fails, stop this mechanism without testing another lesion or tuning lesion strength. No Fly B relearning or teacher-assisted recovery is run in this feasibility session.

Use one opt-in runner and a new exclusive-creation trial-level JSON artifact with this protocol hash. Reproduction must occur in a copy without that artifact. No Git, demo, RPS learning-rule, or historical-artifact edits.
