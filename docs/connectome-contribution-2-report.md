# Study 1 result: the connectome carries the message and lets Fly A learn, but adds nothing beyond a digital message

[Protocol](connectome-contribution-2-protocol.md), locked before any run, with one recorded deviation (below). Simulated Rock-Paper-Scissors only.

**Verdicts under the locked rules**
- **E, endogenous signal: STOPPED.** Fly A's measured neural activity predicted its advice no better than the cue it senses. No natural teaching signal was found, so no teaching test with it was run.
- **A vs B, spikes vs digital: equivalent.** Identical scores on all 15 seeds (difference exactly 0).
- **C, content through the same neural channel:** trained advice beat random advice by +35.07 wins.
- **D, rewired connectome: stopped at teacher readiness.** On all 15 seeds, Fly A built on the rewired connectome could not learn the task (22-25 of 72, about chance), so the teaching comparison could not be run.

## Part E: is there a natural teaching signal?

Five seeds (teacher `35001 + 2s`). For each, a trained Fly A (gate 60-69 of 72) and an untrained Fly A on the same network played the same 144 rounds with identical noise.

**Deviation.** The first run (`data/endogenous_signal/`, kept unchanged) saved empty individual-MBON counts because of a recording bug in the new diagnostic script. The one-line fix and a re-run of the same seeds into `data/endogenous_signal_v2/` were recorded in the protocol before the stop rule was applied. The stop rule itself did not change.

Held-out accuracy predicting the trained Fly A's advice (episodes 3-5; calibrated on 0-2):

| Seed | Cue alone | Individual MBON counts (trained) | Same, untrained Fly A | Pooled MBON group rates (trained) | Same, untrained | Permutation null (95th pct.) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.855 | 0.377 | 0.377 | 0.884 | 0.754 | 0.406 |
| 1 | 0.971 | 0.377 | 0.406 | 0.986 | 0.913 | 0.391 |
| 2 | 0.957 | 0.261 | 0.348 | 0.957 | 0.913 | 0.377 |
| 3 | 0.971 | 0.391 | 0.377 | 0.942 | 0.913 | 0.391 |
| 4 | 0.942 | 0.333 | 0.348 | 0.957 | 0.928 | 0.391 |

Locked stop rule: proceed only if the individual-MBON decoder beat the cue alone by at least 0.10 and beat the untrained-activity decoder by at least 0.10. Observed: -0.59 and -0.02. **Arm E stopped.**

- The pooled group rates, which are what Fly A's own readout reads, predicted the advice about as well as the cue alone (0.945 versus 0.939 on average), so they carry the cue, not extra learned information.
- The individual-MBON decoder was at chance. Its snapshot was taken when each round was logged, which may be after some decision-time counts were reset, so that row may reflect measurement timing rather than a true absence of information. The group-rate row, captured at the decision, supports the same conclusion.
- Trained and untrained Fly A did **not** spike identically: all neurons matched on only 0-11 of 144 rounds per seed. Learning never changes Fly A's synapses (stated in the code), so these differences come from different actions and rewards leading to different carry-over activity, not from stored knowledge. The group rates of the untrained Fly A still predicted the trained Fly A's advice at 0.88.

## Parts A-D: teaching comparison

Rule-switch task as in the advice-budget follow-up (48 rule-B rounds, advice on all 46 observed-cue rounds). 15 fresh seeds (teacher `37001 + 2s`); implementation check seed 20 excluded.

**Real connectome (15/15 completed; teachers 60-70 of 72):**

| Arm | Mean frozen wins at 48 |
|---|---:|
| No advice | 13.07 |
| A. Spike advice (measured DN, decoded) | 65.60 |
| B. Digital advice (same preference, no spikes) | 65.60 |
| C. Random advice through the same spike channel | 30.53 |

Communication: 46/46 messages decoded to the intended action on every seed (100%).

| Contrast at 48 (95% bootstrap, seed 2026111550) | Mean | 95% CI |
|---|---:|---|
| A minus B (spikes minus digital) | 0.00 | 0.00 to 0.00 |
| A minus C (trained minus random content) | +35.07 | +24.93 to +45.07 |

**Rewired connectome (D):** the rewired Fly A scored 22-25 of 72 on all 15 seeds (gate 58; random play scores about 24). Every rewired arm therefore stopped at teacher readiness, and the real-versus-rewired teaching contrast could not be computed.

## What this establishes, and what it does not

- **Established:** in this engineered system, sending advice as spikes and sending it as a number give identical learning; the benefit comes entirely from the teacher's content. The neural channel delivered every message.
- **Established, with a caveat:** Fly A could learn the task on the real connectome and could not on the rewired one. So the real wiring matters for **Fly A's ability to learn from its senses**. The caveat: Fly A's input scaling was fitted on the real wiring, so part of the failure may be a mismatch in input scale rather than wiring alone. This study cannot separate the two.
- **Not established:** any causal role of the connectome in *teaching* or *communication* beyond carrying an externally chosen action. No natural teaching signal was found in Fly A's neurons; the advice is chosen by a software readout and written into Fly B's neurons.
- **Not tested:** whether a learner whose synapses actually change (rather than a software readout) would produce a natural teaching signal.

Files: `data/endogenous_signal/` (first run, recording bug, kept), `data/endogenous_signal_v2/` and `data/endogenous_signal_summary.json` (part E); `data/connectome_contribution_2/seed_0.json`-`seed_14.json`, `smoke_seed_20.json` and `data/connectome_contribution_2_summary.json` (parts A-D). Runners `tools/check_endogenous_signal.mjs`, `tools/check_connectome_contribution_2.mjs`; analysis `tools/analyze_connectome_contribution_2.py`.
