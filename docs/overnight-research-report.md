# Synapse action-preference research progress

## Strongest result

An opt-in **post-sensory hybrid imitation** system passed its locked initial two-student pilot and **two prospectively selected independent teacher–student confirmation pairs** on deterministic Strategy A. A teacher trained from its own RPS rewards encoded its MBON-driven preferred action as simulated DNge175 spikes: 3141 for Rock, 3165 for Paper, both for Scissors. Fly B decoded actual 240–280 ms spike counts and updated its own cue-specific action values from its own game reward plus an auxiliary imitation step. KC→MBON synapses remained frozen. Frozen evaluations had no teacher, exploration, or learning.

| Teacher / student | Trained frozen wins at round 6 | Earliest control to 58/72 | Trained final | Locked pair gate |
| --- | ---: | ---: | ---: | --- |
| 830 / 11007 | 69/72 | round 12 | 69/72 | Pass, initial |
| 830 / 11008 | 66/72 | round 12 | 69/72 | Pass, initial |
| 11009 / 11011 | 69/72 | round 12 | 69/72 | Pass, confirmation 1 |
| 11010 / 11012 | 69/72 | round 12 | 69/72 | Pass, confirmation 2 |

The early mean trained-minus-strongest-control differences at rounds 6/12/24/48 were **+11.75, +11, +8, and +11.75 wins/72**, all above the locked **+4/72** gate. All four trained students finished with cue wins **23/23, 24/24, 22/22**. Shuffled and generic arms sometimes reached the same final ceiling; the result concerns **speed of retained learning**, not final superiority.

## Architecture attempts and failures

1. The first three-action message-first receiver used fixed DN patterns and the old external pooled-MBON cue-recognition gate. Its initial artifact had a **null versus `-1` no-message scoring bug** and is labeled invalid. The corrected write-once run stopped at held-out receiver **11001's 8/12 cue-0 no-message recognition**, below 10/12, despite 12/12 DN decoding and VPN routing. No students trained.
2. A separately locked matched-stream receiver asked whether DN stimulation before the cue changed the actual MBON feature input more than natural neural variability. Development **11003 passed**; held-out **11004 failed** on cue 2 (natural median **0.4953** versus paired code medians **0.8290/0.8705/0.7238**). DN decoding and VPN routing remained 12/12. No students trained; reserved 11005 was not used.
3. The distinct post-sensory architecture placed the pulse after the student had measured its cue and MBON features. Its receiver passed on **11006/11007/11008**: all forced codes decoded 12/12 per cue, VPN routes were 12/12, and pre-pulse features were exactly paired with no-message. This did not revise either earlier failed gate.

## Controls, integrity, and limits

The successful study compared the same student architecture under **trained teacher, no message, genuinely untrained neural teacher, globally shuffled trained messages, and seeded generic neural messages**. All messaged arms used real measured DN spike counts, one opportunity per observed cue, and identical initial student weights, opponent sequences, and keyed streams within a seed. Shuffling preserved the trained source's global action counts while breaking cue alignment; generic messages were pseudorandom and cue-independent. Both passed locked ≤60% per-cue advisory-quality gates. Every student could learn independently to 69/72. Initial and confirmation trained arms decoded **276/276** training messages per student; frozen evaluation used none and changed no weights. Both new confirmation teachers were independently reward-trained and passed source-quality gates. Source-preference **summaries**, rather than every teacher Q vector, are archived; student trial rows contain intended and measured decoded actions.

This is **hybrid simulated neural communication plus an engineered imitation loss**. The student did not receive teacher weights or hidden correct actions, but the decoded teacher-preference action is an explicit learning target held in an engineered spike-count buffer. The study covers one deterministic Strategy A rule and fixed opponent streams. It does not demonstrate biological brain-to-brain teaching, generalization to arbitrary patterns, or the old dopamine-only model learning faster. The earlier one-bit proposal/veto control failure and pooled-decoder failure remain intact. No judge-facing demo or core simulator file was changed.

## Changed files and reproduction

- New isolated runners: `tools/check_rps_action_preference_transfer.mjs`, `tools/check_rps_action_preference_receiver_v2.mjs`, `tools/check_rps_action_preference_receiver_v3.mjs`, `tools/check_rps_action_preference_teaching_v3.mjs`, `tools/check_rps_action_preference_confirmation.mjs`.
- New write-once artifacts: `data/rps_action_preference_transfer.json` (**invalid scoring**), `data/rps_action_preference_transfer_corrected.json` (**failed baseline receiver**), `data/rps_action_preference_receiver_v2.json` (**failed feature stability**), `data/rps_action_preference_receiver_v3.json` (**passed**), `data/rps_action_preference_teaching_v3.json` (**passed initial**), and `data/rps_action_preference_confirmation_{first,second}.json` (**both passed**).
- Read the locked protocols and all per-arm curves in [lab-notebook.md](lab-notebook.md), and the interpretation in [research-story.md](research-story.md). Reproduce **only in a copy lacking each corresponding output artifact**, sequentially: `node tools/check_rps_action_preference_receiver_v3.mjs`; `node tools/check_rps_action_preference_teaching_v3.mjs`; `node tools/check_rps_action_preference_confirmation.mjs first`; `node tools/check_rps_action_preference_confirmation.mjs second`. Do not rerun failed protocols to seek a favorable seed.
- Final regression checks passed: `node --check` on all five new runners, `node tools/check_rps_challenge_demo.mjs`, and `python -m flyhero.test`. These verify syntax and the existing demo/simulator smoke paths; they do not independently validate the new scientific result.

## Next step

The independent [action-preference audit](action-preference-independent-audit.md) found no blocking implementation or arithmetic error in the Strategy A study and documented its provenance and scope limits. A subsequent, separately locked Strategy B generalization is recorded below. Do not tune either successful protocol on its current streams. The live HackPrinceton demo remains unchanged pending a separate claims/UI review.

## Later Strategy B generalization

Existing deterministic Strategy B is fully observable from the previous opponent move. The [locked separate protocol](action-preference-strategy-b-protocol.md) fixed new teacher/student pairs **11013/11015** and **11014/11016**, separate opponent streams, the unchanged post-sensory DN encoding and imitation update, five arms, seven frozen checkpoints, and all gates before training. Teachers learned from their own rewards and scored **67/72** and **69/72** frozen. Both student receivers passed all forced-code and VPN-route gates; every messaged arm decoded **276/276** measured training DN patterns. Independent no-message students reached **70/72** final.

Trained-advice students reached **70/72** and **69/72** disconnected frozen wins after six training rounds, ahead of the earliest control at rounds **24** and **12**. Their early mean advantage over the strongest of Alone, untrained, shuffled, and generic controls was **+17/72** and **+11/72**, exceeding the fixed **+4/72** gate. Both finished **70/72** with **23/23 wins per observed cue**; some controls reached the same final ceiling. The initial and independent confirmation **both passed every locked gate**. Trial-level frozen evaluation rows, full curves, teacher reward-training rows, controller weights, DN counts, and controls are saved in the separate write-once [initial](../data/rps_action_preference_strategy_b_initial.json) and [confirmation](../data/rps_action_preference_strategy_b_confirmation.json) artifacts. Reproduce sequentially only in a copy without them: `node tools/check_rps_action_preference_strategy_b.mjs initial`; `node tools/check_rps_action_preference_strategy_b.mjs confirmation`.

The strongest result now spans **two fixed, deterministic, previous-move-observable RPS rules** under an engineered hybrid imitation objective. It shows faster *retained* learning in the sampled pairs, not higher final accuracy or biological/dopamine-only transfer. The teacher source is a recorded frozen Q-preference tape rather than simultaneous reward-learning gameplay. No noisy, switching, or arbitrary-pattern generalization is established. All old failures remain preserved; no judge-facing demo change was made.
