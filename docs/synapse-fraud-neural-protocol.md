# Fraud-response neural teaching — fixed procedure

This is a separate simulation built on the existing fault-response architecture. It does not alter the original RPS or fault-response datasets. The labels and rewards below are synthetic policy choices, never confirmed fraud outcomes.

## Observable inputs and synthetic policy

The existing fraud backend computes account-history features from each Nessie mock purchase. In priority order, cue 2 is a purchase at least 3x the historical median **and** either a new merchant or at least two purchases in the prior 24 hours; cue 1 is at least two purchases in the prior 24 hours; cue 0 is all other activity. The same rule maps fixture data. These cues are observations, not fraud labels. The live purchase's cue can be sent into a frozen Fly B decision; it is not used as experiment ground truth.

Internal action indices preserve the previous fault-response reward mechanism: `0=BLOCK`, `1=APPROVE`, `2=REVIEW`. Before the controlled incident, the correct actions for cues 0/1/2 are `APPROVE/APPROVE/REVIEW` (`need=[0,0,1]`). After a synthetic coordinated-card-compromise alert, the correct actions are `REVIEW/REVIEW/BLOCK` (`need=[1,1,2]`). For a chosen action, the unchanged `result(action, need[cue])` gives +1 for the declared correct action, 0 for one alternative, and -1 for the other. This is a deliberately broad policy escalation on a small task, not a bank policy recommendation.

## Neural system and controls

Use the existing 6,900-neuron `data/subgraph.json`, `js/sim.js` network, `RpsContextualBandit` teacher, `PreferenceStudent` students, `RpsEpisode`, `preferenceRound`, and `frozenStudentCheckpoint`. Fly A trains on the updated mapping for 12 episodes of 24 rounds. Stop a comparison seed if Fly A scores fewer than 58 of 72 frozen updated-rule trials. Fly B receives four 24-round episodes of shared old-rule history. Then run 96 post-change rounds for four matched arms: alone, trained Fly A advice, random advice, and shuffled trained advice. All arms begin from the same student weights and see matched cue/reward streams.

The existing two descending neurons DNge175 carry the action code `[1,0]`, `[0,1]`, or `[1,1]` as measured spikes in the receiving fly's network. A decoded action, never a direct action argument, drives Fly B's imitation update. Random and shuffled controls also travel through the same DN channel. Count sent and successfully decoded messages; fail on any mismatch. The teacher is absent during frozen checks.

Frozen checks occur after 0, 24, 48, and 96 post-change rounds. Each check contains 72 balanced, shuffled cues (24 of each), with exploration and learning disabled and weights verified unchanged. The predefined recovery threshold is at least 58/72 correct actions. Recovery time is the first checkpoint meeting it, or unreached by 96. The primary comparison is trained Fly B minus the strongest of alone, random, and shuffled at 48 rounds.

## Bounded execution and decision

First run one short end-to-end spike transmission probe (`node tools/run_fraud_neural.mjs probe`), without claiming learning. Next run one focused comparison (`node tools/run_fraud_neural.mjs seed 401`). Only if those complete, run exactly five final seeds 420–424, once each; no seed replacement, parameter sweep, or post-test tuning. Final success requires at least four teacher-gated seeds, mean 48-round advantage at least +5/72 over the strongest per-seed control, and positive advantage in at least four of five completed seeds. Report all individual seeds, failures, recovery times, message counts, and the mean advantage. With five seeds this remains a small synthetic simulation, not a robust estimate of real-world fraud accuracy or biological fly-to-fly teaching.
