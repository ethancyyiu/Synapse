# Study 1: does the fly connectome contribute beyond carrying an engineered message?

Locked before any run of this study. New and separate; the earlier ablation (`connectome-contribution-ablation-protocol.md`), the disruption feasibility test, the spike-loss pilot, and every Science/banking artifact stay unchanged.

## What is already known

- **The message is engineered.** Fly A senses the previous opponent move through its simulated VPN, KC and MBON neurons. A small software readout on top of three pooled MBON rates learns from reward and picks the preferred action. That action is then written into Fly B as spikes on two DN neurons, and a software decoder reads it back.
- **Fly A's learning does not change its neurons.** The code states it: "The selected action alone receives a reward-prediction update; neural synapses stay frozen." All learning is in the software readout.
- **Spikes and digital delivery were identical** in the earlier two-student ablation (same actions, updates and frozen predictions), and every DN message in every study decoded correctly. Fly B's imitation step uses only the decoded symbol, so whenever decoding is perfect the two must agree.
- **Disrupting one MBON group by 50%** changed neural activity but 0 of 69 frozen actions (feasibility test, failed its behavioural gate).
- **An earlier individual-MBON decoder study** did not pass its overall gain gate.

So the open questions are whether Fly A's *natural* activity carries its learned advice, and whether the real wiring matters for teaching or learning at all.

## Part E: is there an endogenous teaching signal? (diagnostic, before any teaching)

**Hypothesis to test (expected to fail, given frozen synapses):** Fly A's measured neural activity carries information about its learned advice beyond the cue it senses.

`tools/check_endogenous_signal.mjs`, seeds 0-4 (teacher `35001 + 2s`; opponent bases 1210000, 1220000, 1230000). For each seed, a trained Fly A (existing procedure, gated at 58/72) and an untrained Fly A on the same network seed play the same 144 rule-B rounds with identical noise. Recorded per round: the cue, the trained Fly A's chosen action (its advice), every neuron's spike count (as a hash) for both, and the individual MBON counts for both.

Analysis (`tools/analyze_connectome_contribution_2.py`): rounds from episodes 0-2 calibrate, episodes 3-5 are held out. Held-out accuracy of predicting the trained Fly A's advice from (i) the cue alone (majority advice per cue), (ii) the trained Fly A's MBON counts (nearest-centroid on standardised log counts), (iii) the untrained Fly A's MBON counts with the same decoder, and (iv) a label-permutation null (decoder (ii) trained on 200 permuted label sets, fixed seed 2026111551). Also: the share of rounds where trained and untrained Fly A spike identically across all neurons.

**Stop rule (fixed now).** Arm E goes on to a teaching test only if, averaged over seeds whose teacher passes its gate, decoder (ii) beats the cue alone by at least 0.10 **and** beats decoder (iii) by at least 0.10, i.e. the activity carries learned information, not just the cue. Otherwise arm E stops and is reported as a negative result: no endogenous teaching signal was found. The teacher's learning rule is not changed to create one.

## Part A-D: teaching comparison

`tools/check_connectome_contribution_2.mjs`. Rule-switch task as in the advice-budget follow-up: rule-A history, then 48 rule-B rounds with advice on all 46 observed-cue rounds, own-action rewards every round, teacher-disconnected 72-game frozen checks at 0, 24 and 48 rounds. Teacher `37001 + 2s`, student `37002 + 2s`; opponent bases 1310000, 1330000, 1350000, 1370000. **Implementation check:** `s = 20` (excluded). **Final:** `s = 0-14`; no seed replaced.

- **A** `neural`: the teacher's preference sent as measured DN spikes and decoded (existing channel).
- **B** `digital`: the same preference applied to Fly B's learner directly, in the same order, with no DN drive (information-matched).
- **C** `random`: a random action through the same spike channel at the same times (a neural channel carrying no teacher information). This is how "simpler neural baseline" is read here: same neurons, same code, no learned content.
- **D** the rewired connectome: teacher and students rebuilt with the simulator's built-in `shuffled` wiring (connections shuffled within each neuron-type block, counts kept; nerve cord untouched), and the same procedure: `aloneS`, `neuralS`, `digitalS`. Frozen checks rebuild the same rewired network. The rewired teacher must pass the same 58/72 gate; if it does not, that seed's rewired arm is a teacher-readiness failure.
- `alone` in each wiring: no advice.

Communication and learning are scored separately: **communication** = share of sent DN messages decoded to the intended action; **learning** = frozen wins.

## Fixed analysis (95% percentile bootstrap, 10,000 paired resamples, fixed seed 2026111550)

1. **Do spikes add anything beyond digital?** `neural - digital` at 48 rounds, real wiring. Spikes add something only if the lower bound exceeds 4; "equivalent" if the whole interval lies within -4 to +4.
2. **Does the teacher's content matter through the neural channel?** `neural - random` at 48 rounds (reported).
3. **Does the real wiring contribute to teaching?** Per seed with both worlds complete: `(neural - alone)_real - (neuralS - aloneS)_rewired` at 48 rounds. The real wiring contributes to teaching only if the lower bound exceeds 4 and communication is at least 99% in both worlds (so the difference is not a delivery failure). If fewer than 10 seeds have a rewired teacher that passes its gate, part 3 is reported as stopped at teacher readiness, together with the teacher scores, since a rewired Fly A that cannot learn is itself evidence about the wiring's role in Fly A's learning, not in teaching.

**Reported, not tested:** every arm's mean at each checkpoint; `alone` versus `aloneS` (does rewiring change learning without a teacher?); `digitalS - neuralS`; teacher gate scores in both wirings.

## Limits

One task, one rule change, one kind of rewiring (degree-preserving within neuron-type blocks). Feature normalisation was fitted on the real wiring, so a rewired network also faces a mismatch in its input scaling; that is part of the disruption. A null result means the real wiring was not needed *in this engineered system*, not that connectomes never matter. The DN channel and the request/advice logic are software-driven, so nothing here tests biological fly-to-fly communication.

## Deviation (recorded before any decision was taken)

The first diagnostic run (`data/endogenous_signal/`, kept unchanged) saved empty individual-MBON counts because of a recording bug: the MBON groups are typed arrays, which `.flat()` does not unpack. Its pooled group rates and spike-identity hashes were recorded correctly. The recorder was fixed (one line) and the same five seeds re-run into `data/endogenous_signal_v2/`; the stop rule is applied to v2. The analysis also reports a second decoder on the three pooled MBON group rates (the exact input Fly A's own readout uses); the stop rule itself is unchanged.
