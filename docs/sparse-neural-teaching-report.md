# Sparse neural action-preference teaching: locked pilot and confirmation

**Result: the sparse teacher-specific claim failed independent confirmation.** Both sparse budgets passed the locked initial Strategy B comparison, but neither passed on the independently fixed teacher/student pair. The 100% reference passed in both. The failed confirmation is preserved alongside the positive pilot; there was no retuning or seed replacement.

The [prospective protocol](sparse-neural-teaching-protocol.md), SHA-256 `D497E9790D34A0AC5EEAD3E51990D039B9C45DF212BD3906F133025A77F724CE`, was written before either result. It fixes Strategy B, teacher/student seeds **18001/18002** and **18003/18004**, opponent streams, 12×24 teacher and student training, seven frozen 72-round checkpoints, send schedules, five arms per budget, +4/72 early mean and earlier-proficiency gates, final reliability gates, and stopping. The confirmation ran because the initial sparse gates passed. Local mutable files and embedded hashes record the intended chronology but are not external preregistration.

## Pathway and implementation validity

The unchanged reward-trained teacher uses simulated VPN/KC/MBON responses to infer a three-action preference. Scheduled messages drive student DNge175 neurons **3141/3165** at **300 Hz, 240–280 ms**, using the fixed codes `(1,0)`, `(0,1)`, `(1,1)`. Student imitation receives only the preference decoded from **measured DN spike counts**, never the teacher target or correct RPS action directly. The student plays its own action before the pulse and updates its cue-head policy from its own game reward at 620 ms; the auxiliary imitation step occurs only when a spike message was decoded. KC→MBON multipliers remain frozen. This is an engineered **hybrid reward-plus-imitation** system, not original dopamine-only learning.

The established `target=null` round path skips DN drive and imitation but keeps the same 680 ms neural schedule, sensory cue, game reward, own-reward update, and per-round keyed random streams. Frozen evaluations copy controller weights into independent simulator instances, disable exploration and all learning, and send no message. The Strategy B opponent's move sequence is independent of the student's selected action. Thus schedules change only message exposure, not training rounds or reward feedback availability.

All prerequisites passed in both stages. Teachers scored **65/72** and **66/72** frozen wins after own-reward training; their source preferences selected the winning action **248/276** and **259/276** training cues, respectively, with all three cue-conditioned source gates above 70%. Each student passed all **144** receiver conditions/probes: 12/12 decoding and VPN routing per cue and forced code, exact matched premessage MBON features, unchanged weights. Silent students began at **24/72** and finished at **70/72** and **71/72**, meeting the independent-learning gate. The synthetic disabled/null imitation checks passed.

The deterministic schedules sent **276/276 (100%)**, **69/276 (25%)**, and **28/276 (10.14%, nominal 10%)** eligible messages in each respective nonsilent arm. All four nonsilent arms at a given budget used the same scheduled slots. The shuffled arm preserved the scheduled trained-action histogram; it changed **184/276, 45/69, 13/28** initial and **181/276, 42/69, 17/28** confirmation positions (see artifacts for exact counts). The generic arm was cue-independent seeded random; the untrained teacher used the same neural inference path with untrained controller weights. The no-message arm was shared. All arms within each stage had identical initial student weights, cue and opponent sequences, 288 training opportunities, random-stream keys, and seven frozen evaluation streams.

Read-only recount of the saved rows verified **2,984/2,984** sent messages across the two stages decoded as intended from nonzero measured DN counts (1,492/stage), and every silent slot had zero DN counts, decoded `-1`, and zero imitation. Every checkpoint stored 72 trial rows whose win count matched the reported count; all had zero DN counts, no imitation, no teaching, and unchanged controller and KC→MBON weights. Student controller weights changed with own-reward training in every arm. No code or scientific artifact from the earlier Strategy A/B studies was changed.

## Frozen disconnected learning curves

Entries are wins/72 after **0, 6, 12, 24, 48, 96, 288** student training rounds. `T` = trained teacher, `U` = untrained teacher, `S` = scheduled shuffled trained preferences, `G` = seeded generic preferences; `Alone` = silence. All nonsilent arms use actual DN spikes.

| Stage | Budget | Arm | 0 | 6 | 12 | 24 | 48 | 96 | 288 | Early mean 6/12/24/48 | First ≥58 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 0 | Alone | 24 | 22 | 24 | 24 | 24 | 70 | 70 | 23.5 | 96 |
| Initial | 100% | T | 24 | 70 | 70 | 70 | 70 | 70 | 70 | 70 | 6 |
| Initial | 100% | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | never |
| Initial | 100% | S | 24 | 47 | 70 | 66 | 70 | 70 | 66 | 63.25 | 12 |
| Initial | 100% | G | 24 | 24 | 24 | 24 | 56 | 68 | 70 | 32 | 96 |
| Initial | 25% | T | 24 | 45 | 46 | 70 | 70 | 70 | 70 | 57.75 | 24 |
| Initial | 25% | U | 24 | 22 | 23 | 24 | 24 | 47 | 70 | 23.25 | 288 |
| Initial | 25% | S | 24 | 3 | 24 | 24 | 47 | 70 | 70 | 24.5 | 96 |
| Initial | 25% | G | 24 | 22 | 24 | 24 | 24 | 70 | 70 | 23.5 | 96 |
| Initial | 10% | T | 24 | 22 | 46 | 70 | 70 | 70 | 70 | 52 | 24 |
| Initial | 10% | U | 24 | 22 | 23 | 24 | 24 | 70 | 70 | 23.25 | 96 |
| Initial | 10% | S | 24 | 22 | 23 | 24 | 47 | 70 | 70 | 29 | 96 |
| Initial | 10% | G | 24 | 22 | 24 | 24 | 47 | 70 | 70 | 29.25 | 96 |
| Confirmation | 0 | Alone | 24 | 23 | 23 | 46 | 46 | 70 | 71 | 34.5 | 96 |
| Confirmation | 100% | T | 24 | 70 | 70 | 70 | 70 | 70 | 71 | 70 | 6 |
| Confirmation | 100% | U | 24 | 23 | 23 | 23 | 23 | 23 | 24 | 23 | never |
| Confirmation | 100% | S | 24 | 31 | 70 | 64 | 70 | 70 | 71 | 58.75 | 12 |
| Confirmation | 100% | G | 24 | 23 | 23 | 46 | 46 | 70 | 70 | 34.5 | 96 |
| Confirmation | 25% | T | 24 | 47 | 70 | 70 | 70 | 70 | 71 | 64.25 | 12 |
| Confirmation | 25% | U | 24 | 23 | 23 | 46 | 46 | 70 | 71 | 34.5 | 96 |
| Confirmation | 25% | S | 24 | 47 | 70 | 70 | 70 | 70 | 71 | 64.25 | 12 |
| Confirmation | 25% | G | 24 | 23 | 23 | 46 | 46 | 70 | 71 | 34.5 | 96 |
| Confirmation | 10% | T | 24 | 23 | 47 | 70 | 70 | 70 | 71 | 52.5 | 24 |
| Confirmation | 10% | U | 24 | 23 | 23 | 46 | 46 | 70 | 71 | 34.5 | 96 |
| Confirmation | 10% | S | 24 | 23 | 47 | 70 | 70 | 70 | 71 | 52.5 | 24 |
| Confirmation | 10% | G | 24 | 23 | 47 | 70 | 70 | 70 | 71 | 52.5 | 24 |

## Locked gates and interpretation

| Budget | Initial result | Independent confirmation | Reason |
| --- | --- | --- | --- |
| 100% | **Pass** | **Pass** | Early mean beat every control by ≥4/72; first ≥58 at round 6 before controls; final cue/action gates passed. This is a reference replication under new seeds. |
| 25% | **Pass** | **Fail** | Confirmation trained and shuffled arms tied at **64.25/72** early mean and both first reached 58 at round 12. Final trained 71/72 and all cue/action gates passed, but teacher-specific early advantage and earlier proficiency did not. |
| 10% | **Pass** | **Fail** | Confirmation trained, shuffled, and generic arms tied at **52.5/72** early mean and first reached 58 at round 24. Final trained 71/72 and all cue/action gates passed, but teacher-specific early advantage and earlier proficiency did not. |

The trained arm's early mean declined from **70 → 57.75 → 52** in the initial pair and **70 → 64.25 → 52.5** in confirmation as exposure fell from 100% to 25% to 10%. Nevertheless all trained arms retained **70/72** or **71/72** by round 288 with the teacher disconnected. This is retained independent student performance; it does **not** establish a sparse teacher-specific benefit because the confirmation controls matched the sparse arms' early curves and final scores. In the confirmation, trained preferences were far more often winning on scheduled rounds than shuffled/generic preferences (25%: **66/69** versus **26/69, 27/69**; 10%: **27/28** versus **12/28, 10/28**). The observed early ties therefore cannot be explained simply by a failed source or DN decoder. Few helpful messages can suffice on this deterministic three-cue task, and control messages can also happen to produce them; the experiment does not isolate which individual update caused each tie.

This study supports the limited claim that **28 or 69 correctly decoded simulated spike messages can coexist with a retained three-action policy in these two student seeds**, and that the initial sparse pair showed an early trained-advice advantage. It does **not** establish that reduced-frequency trained advice reliably beats exposure-matched controls. It measures message count, not physical bandwidth, energy, or real-world cost. The earlier direct-digital ablation found identical learning behavior and remains the relevant limit on claims of neural-channel superiority. The controller, decoder, and imitation update are engineered; biological transfer and dopamine-only teacher-specific learning remain unproven.

## Artifacts and reproduction

- [Initial write-once trial rows](../data/rps_sparse_neural_teaching_initial.json), SHA-256 `3EC721080AD0EF45C1C32EDE159E03C4FB904E273DF12F2724D4B68AB1D056CD`.
- [Confirmation write-once trial rows](../data/rps_sparse_neural_teaching_confirmation.json), SHA-256 `4D7643872745491BB49A097CA4F8579F5F6CDF5738D03C4A8DC204D4BDCE3103`.

Each artifact embeds the protocol hash, teacher training/frozen rows, receiver probes, source tape summaries, per-budget schedules and control quality, all 288 student training rows/arm, all seven 72-row disconnected frozen evaluations/arm, controller snapshots, and gate outcomes. `node --check tools/check_rps_sparse_neural_teaching.mjs` passed. Reproduce **only in a copy without the write-once files**: `node tools/check_rps_sparse_neural_teaching.mjs initial`; after its sparse pass, `node tools/check_rps_sparse_neural_teaching.mjs confirmation`. No further experiment was run.
