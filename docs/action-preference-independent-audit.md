# Independent audit of action-preference teaching

> Archive note: Calendar dates were removed from this working tree after the audit. The hashes and chronology statements below refer to the original research files; they do not authenticate the edited files.

## Scope and verdict

This was a **read-only audit** of the initial pilot and two confirmation artifacts, their isolated runners, and the recorded protocol. No simulation, training, threshold change, demo edit, or Git operation was performed. **No blocking implementation or arithmetic error was found.** The saved trials support faster *frozen, teacher-disconnected* learning from trained advice than from each recorded control in all four students. **Recommendation: requires further validation before using this as a broad main research claim.** It can be shown as a narrowly labeled archived result for the engineered hybrid imitation controller on the fixed deterministic Strategy A task.

The three audited write-once files are [`rps_action_preference_teaching_v3.json`](../data/rps_action_preference_teaching_v3.json) (SHA-256 `EA8A4CCF86FD5305365778CD921CBEAC650800D7436C81FA18C0F4ABFBA0FFF3`), [`rps_action_preference_confirmation_first.json`](../data/rps_action_preference_confirmation_first.json) (`B6954E33ACE7A28A933AC5569CDDCF32A2B65F4E930719F8C4781D4DF3493DEA`), and [`rps_action_preference_confirmation_second.json`](../data/rps_action_preference_confirmation_second.json) (`7E7570E7E6028440D8ED468263A40274E29709B86DB2E00429F973DA2E85127B`).

## Ten checks

| # | Assessment | Evidence and limit |
| ---: | --- | --- |
| 1 | **Partial** | The [lab notebook](lab-notebook.md) records the v3 receiver, initial pilot, five arms, seven checkpoints, numerical gates, and later two confirmation pairs before their reported results. Runner constants and artifact protocols agree. Mutable documents and result files cannot independently prove the time order of locking versus execution; there is no immutable preregistration record. |
| 2 | **Pass by code; artifact limited** | `trainTeacher()` in the [pilot](../tools/check_rps_action_preference_teaching_v3.mjs) and [confirmation](../tools/check_rps_action_preference_confirmation.mjs) runners uses 12×24 `RpsEpisode` rounds with `learn: true`, the ordinary Strategy A opponent, reward calls, and the existing contextual-bandit update. The source then freezes those weights and takes the argmax of MBON-driven Q scores. The winning action is calculated only later for quality analysis. The new artifacts store teacher frozen aggregates (61/72, 60/72, 63/72), not the teacher's training rows or complete Q trajectory, so independent teacher-training reconstruction was outside this read-only audit. |
| 3 | **Pass** | Fixed preferred-action codes are DN3141 only, DN3165 only, or both. The runners stimulate at 300 Hz during 240–280 ms, count actual `net.spikes`, and decode the two counts. Across all four students, each messaged arm had 276/276 training messages with positive intended and zero unintended DN counts; fresh confirmation receiver probes also passed every forced-code condition. A separate spike-count buffer is engineered, so this is not persistent neural memory. |
| 4 | **Pass** | `studentRound()` passes the intended action only to `setDrive`; its 620 ms `imitate()` call receives `row.decoded`, computed from measured counts. Replaying every saved selected-action reward step and decoded-action softmax step analytically from the trial rows reproduced all **20 final student controller parameter sets exactly** (maximum absolute parameter difference 0). No hidden correct-action label or teacher weights enter that update. |
| 5 | **Pass with exposure qualification** | Within each student, all five arms have byte-identical starting controller weights, identical 288 opponent cue/move sequences, identical pre-message three-MBON features, the same keyed random streams, and seven 72-round evaluation checkpoints. All arms get own-reward updates; the four messaged arms each get 276 decoded messages and imitation opportunities, while Alone intentionally gets none. Shuffling preserves the trained source's global action counts exactly. Generic matches message count and timing, but has a different action histogram; untrained advice always selects Rock because its Q values are tied at zero. |
| 6 | **Qualified pass** | The shuffled tape is a fixed global permutation of trained preferences; the generic tape uses a fixed cue-independent RNG. Neither generator reads the correct action. From saved training rows, shuffled preferences would win **103/276** in the pilot and **100/276** in each confirmation; generic would win **83/276** in all three. Every shuffled and generic per-cue rate stays below the locked 60% ceiling. These messages still coincide with the right action on some trials; the gate rules out a strong accidental oracle, not every finite-sample correlation. The fixed quality gate also conditions continuation on controls being weak enough, although no resampling occurred. |
| 7 | **Pass** | Recalculated early means at rounds 6/12/24/48 and the first sampled 58/72 checkpoint from the saved checkpoint arrays. All **140** reported checkpoint wins equal the sum of their three saved cue-win subtotals. Trained advice beat the strongest control's early mean by **11.75, 11, 8, and 11.75 wins/72** for students 11007, 11008, 11011, and 11012. Each cleared the locked +4/72 margin and reached 58/72 at round 6; the earliest control in each case reached it at round 12. All four trained arms ended 69/72, with observed-cue wins 23/23, 24/24, 22/22. |
| 8 | **Pass by code; artifact limited** | `frozen()` creates a separate student/network copy, sets `explore=false` and `readoutLearn=false`, sends `target=null`, calls the round with `learn=false`, and asserts unchanged controller weights, zero DN counts, and zero imitation. Every saved checkpoint reports disconnected and weights unchanged. Frozen evaluations still simulate the ordinary reward and dopamine response, but their learning flag prevents KC→MBON and controller changes. Artifacts store checkpoint aggregates rather than individual frozen trial actions, so those 72 outcomes cannot be independently reconstructed from raw evaluation rows alone. |
| 9 | **Partial** | The notebook lists fresh teacher/student pairs 11009/11011 and 11010/11012 and a stop-after-first-failure rule before the reported confirmations. The runner hardcodes them, requires a passed initial artifact, and refuses the second stage unless the first artifact passed. No file provides an immutable timestamped seed commitment, so retrospective seed choice cannot be disproved from the archive alone. There was no recorded replacement or parameter change between the two confirmation runs. |
| 10 | **Pass** | The invalid first scoring artifact, corrected message-first receiver failure, v2 held-out feature-stability failure, earlier proposal/veto teaching failure, and noisy pooled-decoder failures remain separate and explicitly failed in the [lab notebook](lab-notebook.md), [research story](research-story.md), and [overnight report](overnight-research-report.md). The successful post-sensory protocol has different timing and an added imitation objective; its pass does not revise any older gate. |

## Complete frozen win curves

Each cell is wins/72 after the indicated **number of student training rounds**. All checkpoints used the same 51000–51002 held-out opponent streams, without teacher messages, exploration, or learning. `T` = trained teacher, `A` = Alone, `U` = untrained neural teacher, `S` = globally shuffled trained advice, `G` = seeded generic neural advice.

| Student | Arm | 0 | 6 | 12 | 24 | 48 | 96 | 288 |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11007 | T | 24 | 69 | 69 | 69 | 69 | 69 | 69 |
| 11007 | A | 24 | 47 | 47 | 47 | 47 | 47 | 69 |
| 11007 | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| 11007 | S | 24 | 22 | 69 | 69 | 69 | 69 | 65 |
| 11007 | G | 24 | 0 | 47 | 47 | 48 | 69 | 45 |
| 11008 | T | 24 | 66 | 69 | 69 | 69 | 69 | 69 |
| 11008 | A | 24 | 24 | 46 | 46 | 46 | 69 | 69 |
| 11008 | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| 11008 | S | 24 | 22 | 69 | 69 | 69 | 69 | 69 |
| 11008 | G | 24 | 0 | 53 | 47 | 48 | 69 | 45 |
| 11011 | T | 24 | 69 | 67 | 69 | 69 | 69 | 69 |
| 11011 | A | 24 | 24 | 24 | 24 | 24 | 46 | 69 |
| 11011 | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| 11011 | S | 24 | 45 | 66 | 65 | 66 | 66 | 69 |
| 11011 | G | 24 | 0 | 42 | 45 | 67 | 69 | 69 |
| 11012 | T | 24 | 69 | 69 | 69 | 69 | 69 | 69 |
| 11012 | A | 24 | 24 | 46 | 69 | 69 | 69 | 69 |
| 11012 | U | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| 11012 | S | 24 | 22 | 69 | 69 | 69 | 69 | 68 |
| 11012 | G | 24 | 0 | 69 | 69 | 69 | 69 | 66 |

## Interpretation and recommendation

The strongest defensible statement is: **In this opt-in hybrid simulated neural system, pretrained Fly A action preferences transmitted as measured DN spikes accelerated Fly B's retained, disconnected RPS learning relative to four controls on the fixed deterministic Strategy A protocol for two pilot students and two independent confirmation students.** The message changes a later imitation update, **not the action played in the same round**. Fly B also learns from its own reward, and KC→MBON synapses remain frozen. This is engineered machine-learning transfer through simulated neural activity, not biological fly-to-fly teaching or a success of the original dopamine-only policy.

The result is narrow: one opponent rule, the same fixed training/evaluation opponent streams for every cohort, small seed count, a near-perfect teacher preference tape, and a generic control matched on exposure but not exact action frequencies. Controls sometimes reach the same final ceiling; the effect is early speed, not final superiority. The saved frozen checkpoints lack trial rows and the teacher's learning/Q trajectory is summarized, limiting independent reconstruction. Prospective timing is documented but not independently certified. **Requires further validation for a headline claim of general teacher-specific neural teaching.** A demo may cite the exact four-student archived comparison and show the actual spike-count interface, provided it labels the hybrid imitation objective and these limits; no demo integration was performed in this audit.

During message-tape generation, `teacherTape()` keeps Fly A's trained Q parameters and neural state, but advances the Strategy A opponent with `game.play(0)` and does **not** feed that round's reward or dopamine into Fly A. Thus this is frozen teacher inference after reward training, not a teacher continuing to play and learn alongside Fly B. The teacher preferences are recorded ahead of student training; Fly B then receives their stimulation through its own simulated DN spikes. Strategy A depends on the opponent's previous move, so the dummy player action does not determine its next move; the same construction would need review for an opponent that responds to the player's action. The archived teacher's separate 61/72, 60/72, and 63/72 frozen game evaluations use a different 400 ms episode schedule from the 680 ms message tape, so those scores should not be described as message-tape action accuracy. The untrained teacher's all-Rock advice is correct on all 88 cue-1 training observations, so that control is not uninformative on every cue.
