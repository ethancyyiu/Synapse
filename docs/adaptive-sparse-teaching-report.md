# Student-requested sparse RPS teaching: locked initial result

**Result: valid pilot, failed teacher-specific gate.** The student-requested trained-message arm learned faster than an evenly spaced 25% trained-message arm and silence, but did **not** beat its exposure-matched shuffled control by the predeclared **4/72** early frozen-win margin or reach proficiency earlier. The independent confirmation was therefore **not run**. The earlier fixed-schedule 25%/10% confirmation failure remains unchanged.

The [prospective protocol](adaptive-sparse-teaching-protocol.md), SHA-256 `F8BB0CC24E443E9BE588025231BDD8FA44945F95EE2DE4BE9BCDA82C62783EC8`, fixed Strategy B, teacher/student **18101/18102**, conditional independent **18103/18104**, 12×24 teacher/student budgets, opponent/evaluation streams, an experience-count request rule, a hard **69/276** DN-message cap, six arms, seven frozen checkpoints, and the +4/72 and earlier-proficiency gates before simulation. The write-once [initial artifact](../data/rps_adaptive_sparse_teaching_initial.json), SHA-256 `AB2DAA71CF5713634E9053D2DC107460A0DB4D9B9497A00C3C346F1DFE3E6DE9`, embeds the matching protocol hash. Local mutable files and hashes do not provide external proof of preregistration timing.

## Mechanism and validity

Fly B used the **count of its own prior completed reward observations for the currently VPN-routed cue** as a simple uncertainty proxy. At 240 ms, after its own provisional action but before the current opponent move/reward at 280 ms, it requested advice if that cue had fewer than **23** prior reward observations and the **69-message** cap was not exhausted. This count is not a calibrated uncertainty probability and does not use reward values, teacher scores, the hidden next move, or the correct action. It causes front-loaded, cue-balanced requests. Experience counts change only the request decision, never the student's action, reward, imitation, or frozen evaluation equations.

Because Fly A cannot read Fly B's private counts, the opt-in runner modeled a **one-bit digital request to a teacher-side transmitter on each of 276 observed-cue rounds**. This reverse channel is outside the simulated connectome; it tells the transmitter only request/no-request. It is not a demonstrated neural return pathway. When requested, a frozen, own-reward-trained Fly A preference was encoded in DNge175 3141/3165 spikes at the unchanged 300 Hz, 240–280 ms pulse. Fly B's imitation step used only the preference decoded from the **actual measured DN spike counts**. A silent round kept the same 680 ms simulation, its own RPS reward update, and keyed random streams, with zero DN drive and imitation. The teacher source was a prerecorded frozen neural-inference tape, as in the validated Strategy B architecture. KC→MBON weights stayed frozen.

All prerequisites passed. Teacher 18101 scored **65/72** frozen wins after learning from its own rewards (cue wins **20/23, 21/23, 23/23**); its frozen preference tape selected the winning action **84/92, 73/90, 94/94** times by cue. Student 18102 passed all **144** receiver probes, including every forced code and cue route at **12/12**; no-message student learning went **24→70/72**. The unchanged imitation-disabled/null checks passed. The original source, receiver, and selected-action reward-learning functions were reused; syntax and static comparison checks found no learning-equation change.

The four adaptive arms independently generated the **same request trace** from their own prior cue-observation counts, without copying another arm's actions or rewards. Each sent **276 upstream request bits** (including zeros) and **69 downstream DN messages**, exactly **23 per cue**; the fixed trained arm sent **69 DN messages** and no request bits. No budget was left unused. Shuffling preserved the scheduled trained-preference action histogram and moved **48/69** positions. On the adaptive slots, trained preferences would win **60/69**, shuffled **22/69**, untrained **23/69**, and generic **32/69**; these correctness counts were measured for analysis, not used to choose schedules. Every one of the **345/345** delivered DN messages across the five nonsilent arms decoded as its intended action from measured spikes. All silent slots had zero DN counts, decoded `-1`, and zero imitation while own-reward updates continued. All arms shared initial student weights, the 288 cue/opponent sequence, keyed streams, and seven identical frozen evaluation streams. Frozen rows contained no requests, messages, exploration, imitation, or learning; controller and KC→MBON weights remained unchanged during evaluation.

## Frozen disconnected learning curves

Values are wins/72 after **0, 6, 12, 24, 48, 96, 288** student training rounds. Every checkpoint evaluates a separate frozen student with Fly A disconnected. Online training wins are a different measure.

| Arm | 0 | 6 | 12 | 24 | 48 | 96 | 288 | Early mean 6/12/24/48 | First ≥58 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive trained | 24 | 47 | 70 | 70 | 70 | 70 | 70 | **64.25** | 12 |
| Fixed 25% trained | 24 | 24 | 47 | 70 | 70 | 70 | 70 | 52.75 | 24 |
| Adaptive shuffled | 24 | 45 | 67 | 68 | 70 | 70 | 70 | **62.5** | 12 |
| Adaptive untrained | 24 | 24 | 24 | 24 | 24 | 47 | 70 | 24 | 288 |
| Adaptive generic | 24 | 11 | 41 | 70 | 70 | 70 | 70 | 48 | 24 |
| Alone | 24 | 15 | 24 | 47 | 70 | 70 | 70 | 39 | 48 |

| Training round | 0 | 6 | 12 | 24 | 48 | 96 | 288 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cumulative adaptive DN messages | 0 | 5 | 11 | 23 | **46** | 69 | 69 |
| Cumulative fixed DN messages | 0 | 2 | 3 | 6 | **12** | 23 | 69 |

The adaptive trained arm exceeded fixed trained by **11.5/72** early mean and reached the sampled 58/72 threshold at round **12** rather than **24**. This comparison also has much greater **early** message exposure under the adaptive schedule (46 versus 12 by round 48), despite equal 69-message totals. It does not isolate a superior timing rule from simply moving message opportunities earlier. The exposure-matched adaptive shuffled arm reached the threshold at the **same round 12** and was only **1.75/72** below trained on the early mean; both differences fail the locked teacher-specific gate. Adaptive trained exceeded untrained, generic, and silence by **40.25, 16.25, and 25.25/72**, respectively, but one positive subset cannot rescue a failed all-control comparison.

Every arm finished at **70/72** disconnected. The trained adaptive arm had **23/23** wins for each observed cue in its final frozen evaluation and selected all three actions at least 12 times, so its final reliability gate passed. There was **no retained final-score advantage** over controls. The shuffled source had substantially less action information than trained advice yet produced similar early performance; the saved result does not identify the exact causal subset of useful messages. The experience-count rule is predictable from public cue occurrences and does not respond to confidence or error magnitude, a key limitation of calling it adaptive uncertainty. The requested confirmation pair was not simulated because the initial +4/72 and strictly-earlier-than-every-control gates failed.

## Conclusion and reproduction

This pilot establishes that the existing engineered hybrid system can transmit and use **69 measured DN messages** while the student retains a strong disconnected RPS policy. It does **not** establish a teacher-specific retained-learning benefit from the adaptive 25% schedule, nor a communication-efficiency advantage over fixed timing. Each adaptive arm also used **276 digital upstream request bits**; no physical bandwidth, energy, or deployment cost was measured. The earlier matched direct-digital ablation found identical learning behavior, and this study adds no evidence of neural superiority or biological fly-to-fly teaching.

**Stop this adaptive-timing branch under the current rule.** The scientifically supported demonstration remains the separately validated full-message Strategy A/B comparison, labeled as hybrid simulated neural communication. No further adaptive schedule or threshold was tested here.

The write-once artifact stores teacher training/frozen rows, 144 receiver probes, request mask and per-round request times, actual DN counts/decodes, all six 288-round training rows, seven 72-row frozen evaluations per arm, controller snapshots, message counts, and every gate. Reproduce **only in a copy lacking the artifact**: `node tools/check_rps_adaptive_sparse_teaching.mjs initial`. `node --check tools/check_rps_adaptive_sparse_teaching.mjs` passed. No confirmation command was executed, and no demo or historical result was modified.
