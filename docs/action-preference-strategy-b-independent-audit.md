# Independent Strategy B action-preference audit

> Archive note: Calendar dates were removed from this working tree after the audit. The hashes and chronology statements below refer to the original research files; they do not authenticate the edited files.

## Verdict and scope

**No blocking arithmetic or implementation inconsistency found.** Read-only checks of the [initial](../data/rps_action_preference_strategy_b_initial.json) and [confirmation](../data/rps_action_preference_strategy_b_confirmation.json) trial-level artifacts support the locked *early, retained* learning-speed result for this engineered hybrid system. It is ready for a narrowly labeled archived demo comparison of two fixed observable rules. This audit did not simulate, train, alter thresholds, or validate arbitrary opponents or biological teaching.

The initial and confirmation artifact SHA-256 hashes are `2EB12B676DDCA6185DBC31630E60197B1728D489120C54F00F59E2E48A3A50CC` and `E79405A2A82C063D4051B99EFE4B136580E252831D5A51EEB1C1D0ACAA82DF92`. Both embed the hash `87DBBC6E4AF6630887392EA86951AC3D6F8ECDF3D748B9CCBFF5416A5064BB3E` of the [locked protocol](action-preference-strategy-b-protocol.md). Local mutable files do not independently certify when it was written.

## Ten requested checks

| # | Status | Evidence and qualification |
| ---: | --- | --- |
| 1 | **Pass by code and saved rows** | `trainTeacher()` uses 288 ordinary Strategy B game rounds with `learn:true`; all 288 recorded training rewards per teacher equal the RPS outcome of its own action and opponent move. The existing contextual bandit receives that reward. No strategy identity or correct action enters its policy. Teacher frozen wins were 67/72 and 69/72. The artifacts do not store a complete teacher weight trajectory. |
| 2 | **Qualified pass** | `teacherCopy()` loads the trained controller, disables exploration and readout learning, and `teacherTape()` chooses the argmax of its simulated MBON-driven Q scores. Its weights stay fixed. Source-quality aggregates are 86/93, 82/90, 93/93 and 83/93, 87/91, 92/92 by cue. Full per-trial teacher-tape rates and Q scores were not saved, so this part is verified from runner code and source aggregates, not independently replayed from tape rows. |
| 3 | **Pass** | The 144 receiver probes per student decode every forced three-action DN pattern, with intended spikes and zero off-target counts. For each stage, all four messaged arms have 276/276 training rows whose decoded target matches measured DN3141/3165 spike counts: 1,104/1,104 arm-round transmissions per stage. The no-message arm has no DN spikes. |
| 4 | **Pass** | The student action is chosen before the pulse. Its reward update uses its own action and game reward; `imitate()` receives `row.decoded`, never the source `target` or a winning-action label. Independent arithmetic replay of every own-reward and decoded-action softmax imitation step reproduces all ten saved final student controllers with maximum absolute weight difference below `9e-16`. The intended target is used to select DN stimulation and for checks, not as the update argument. |
| 5 | **Pass with control qualification** | Within each pair, five arms have identical initial controller weights, 288 cue/opponent sequences, pre-message MBON feature arrays, keyed random streams, and seven identical 72-round evaluation schedules. All have own-reward learning; four receive 276 neural messages and imitation opportunities, while silence intentionally has none. Shuffling preserves the trained tape's action histogram. Generic messages match exposure but not action histogram; untrained Q ties yield Rock advice. |
| 6 | **Pass** | All 5,040 saved frozen rows across both stages have zero DN counts, decoded sentinel `-1`, zero reward and imitation steps, and actions equal greedy score argmax. Each of 70 checkpoints records `disconnected:true` and `weightsUnchanged:true`; runner creates separate frozen simulator/controller copies with exploration and learning disabled. |
| 7 | **Pass** | Trained arms first exceed 58/72 at sampled round 6: 70/72 initial and 69/72 confirmation. Earliest controls do so at round 24 and round 12 respectively. |
| 8 | **Pass** | Recomputed mean wins at rounds 6, 12, 24, 48: trained 70 versus strongest control 53 initially (**+17/72**); trained 69.75 versus strongest control 58.75 in confirmation (**+11/72**). Both exceed +4/72 against *every* control. |
| 9 | **Pass** | Both trained students finish 70/72 disconnected. On the 69 observed-cue rounds each scores 23/23 on each cue and selects each required action 23 times. Their no-message counterparts also finish 70/72, so this is a speed result, not a final-accuracy advantage. |
| 10 | **Partial: chronology not independently provable** | The separate protocol names both seed pairs, streams, budgets, controls, gates, and confirmation stop rule; both artifacts embed its matching hash. The runner refuses to overwrite results and checks that the initial stage passed before confirmation. Mutable local records and SHA hashes do not provide an independently timestamped preregistration or rule out prior exploratory choice. No parameter or seed substitution is recorded between stages. |

## Recalculated frozen five-arm curves

Each cell is **wins/72** on an independent, frozen, teacher-disconnected evaluation after the stated number of student training rounds. `T` is trained teacher, `A` no advice, `U` untrained teacher, `S` globally shuffled trained advice, and `G` seeded generic advice. Online training wins are a different measurement.

| Stage / student | Arm | 0 | 6 | 12 | 24 | 48 | 96 | 288 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial 11015 | T | 24 | 70 | 70 | 70 | 70 | 70 | 70 |
| Initial 11015 | A | 24 | 24 | 24 | 47 | 70 | 70 | 70 |
| Initial 11015 | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| Initial 11015 | S | 24 | 13 | 47 | 70 | 70 | 70 | 70 |
| Initial 11015 | G | 24 | 36 | 47 | 60 | 69 | 70 | 70 |
| Confirmation 11016 | T | 24 | 69 | 70 | 70 | 70 | 70 | 70 |
| Confirmation 11016 | A | 24 | 23 | 24 | 47 | 47 | 70 | 70 |
| Confirmation 11016 | U | 24 | 23 | 24 | 24 | 24 | 24 | 24 |
| Confirmation 11016 | S | 24 | 24 | 23 | 70 | 69 | 69 | 70 |
| Confirmation 11016 | G | 24 | 25 | 70 | 70 | 70 | 69 | 60 |

## Locked gates and limits

| Gate | Initial | Confirmation |
| --- | --- | --- |
| Teacher: ≥58/72, ≥70% per cue, ≥12 choices/action, synapses fixed | **Pass:** 67/72; 22/23, 21/23, 23/23; 24/22/23 | **Pass:** 69/72; 22/23, 23/23, 23/23; 22/23/24 |
| Receiver: forced code, VPN route, paired pre-pulse features, integrity | **Pass:** 12/12 in every cue/code branch | **Pass:** 12/12 in every cue/code branch |
| Synthetic imitation and no-message independent learning | **Pass:** 24→70/72 | **Pass:** 24→70/72 |
| Source quality, all three actions, shuffled/generic ≤60% per cue, shuffle ≥50% changed | **Pass:** 195/276 shuffled positions changed | **Pass:** 199/276 changed |
| Equal exposure, frozen evaluation, neural weight integrity | **Pass** | **Pass** |
| Trained first 58/72 by round 48 and before each control | **Pass:** 6 vs earliest control 24 | **Pass:** 6 vs earliest control 12 |
| Early mean ≥4/72 above each control; final ≥58/72, ≥70% per cue, ≥12/action | **Pass:** +17 over strongest control; final 70/72 | **Pass:** +11; final 70/72 |

The old pooled-MBON external decoder scored only 9/12 on initial student cue 0, including without a message. It was not the unchanged VPN-derived cue router used by this new protocol; its historical failed gate stays failed. The teacher source is a **prerecorded frozen inference tape** after own-reward training, not a teacher that continues updating during student training. The simulated DN spike counter and auxiliary decoded-action imitation step are engineered; KC→MBON synapses stay frozen. There are two Strategy B teacher/student pairs and fixed deterministic opponent rules only. Generic advice matches exposure but not action histogram, and controls can reach the same final ceiling. The earlier failed proposal/veto, noisy receiver, and message-first experiments remain separate negative findings.

**Presentation recommendation:** show the exact archived five-arm curves and early speed advantage with their controls, label the system *hybrid simulated neural communication*, and keep live two-student Challenge results separate from this research evidence.

## Read-only reproduction

From the repository root, parse the two Strategy B JSON artifacts and sum `pilot.byArm[arm].checkpoints[i].rows[*].reward === 1` for each frozen checkpoint. Recompute early means over rounds 6/12/24/48 and first sampled `wins >= 58`. Replay training rows in order with the selected-action update `0.03 × (reward − scores[action]) × [features, 1]`, followed by the decoded-target softmax imitation gradient `0.03 × (oneHot(decoded) − softmax(updatedScores)) × [features, 1]`. This audit performed those read-only calculations and compared final weights within floating-point precision; it did not invoke the write-once training runner.
