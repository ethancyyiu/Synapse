# Earlier RPS experiment (history, superseded)

This page preserves the first Rock-Paper-Scissors teaching experiment and its follow-up diagnostics, moved here from the README. **It found no evidence that the teacher helped.** It used a different channel (three intermediate neurons, quantized spike counts) and the original dopamine-only learner. The current result uses a different protocol; see the [README](../README.md) and the [rule-switch cohort protocol](rule-switch-cohort-protocol.md). The legacy page (`legacy.html`) still shows this experiment.

Commands below run from the repository root.

## What is simulated



`data/subgraph.json` contains 6,900 neurons and 990,102 aggregate connections representing 6,419,481 synapses from the Janelia/Google male CNS connectome. `flyhero/sim.py` is the Python reference: a leaky integrate-and-fire simulation at 1 ms steps. `js/sim.js` is its browser port. Each fly owns its own membrane voltages, spike traces, random stream, and synaptic multipliers. Only KC→MBON multipliers learn. The original `BioRL` reward-prediction-error rule drives PAM or PPL1 dopamine neurons, then changes the chosen action group's recently active KC→MBON synapses. The RPS learner reuses this rule with a mandatory three-move readout.

The opponent supports a repeating sequence, a rule based on its previous move, a rule based on the player's previous move, and a rule that switches after a chosen round. Each strategy can optionally make random deviations. The selected experiment uses `previous_opponent`, offset 1, and 10% random deviation. The fly observes the previous opponent move. It receives neither the current move nor the strategy parameters. The environment also records previous player move and outcome for future input-encoding studies; the present encoding uses only previous opponent move, through the three existing visual-neuron groups.

## Communication protocol

The channel samples **three intermediate neurons downstream of MBONs at 180 ms** of each round, before the teacher chooses a move or sees the opponent's current move. The cells are IDs `2709`, `2360`, and `2325`. They have strong anatomical input from MBONs and each fired on at least 13 of 24 rounds in a frozen, untrained calibration. This selection uses no RPS rewards or hidden-rule labels; it is **not** a known fly-to-fly neural pathway. Unlike upstream Kenyon cells, these cells can reflect changes to KC→MBON weights. Each spike count since 50 ms is capped at two and sent as one of three values (0, 1, 2). The student receives those values as 0, 75, or 150 Hz drive on two visual neurons per channel until 240 ms: `(0,3)`, `(1,4)`, and `(2,5)`. These six target neurons are reserved for the channel in every condition, so a zero signal cannot suppress the normal observation. Bandwidth is configurable from one to three values per round. Each delivered vector is logged in the corresponding run in `data/rps_results.json`.

The channel handles only integer spike counts. It cannot read or transmit teacher synaptic multipliers, the teacher's chosen move, the opponent's current move, or the opponent strategy. Student training and evaluation use separate network instances. In final evaluation the channel is disconnected and synaptic learning is frozen.

## Controlled comparison

`tools/run_rps_experiments.mjs` runs three fixed seeds, 12 episodes per student, and 24 rounds per episode. All four main student groups receive the same opponent strategy, initial student seed, training budget, and held-out evaluation protocol:

| Group | Training input |
|---|---|
| Alone | No teacher signal |
| Trained | Spike counts from a teacher pretrained for 12 episodes |
| Untrained | Spike counts from an untrained, frozen teacher |
| Shuffled signal | The trained teacher's spike-count vectors permuted within each episode |

The trained and untrained signal sources use matched random-stream initializations during collection; the trained source retains its learned synapses. An additional alone group uses the repository's shuffled-connectome wiring control. The experiment reports each training curve, the last three episodes, rounds to a 60% rolling three-episode win-rate threshold, and win rate on three held-out opponent sequences per seed with the teacher disconnected. The 10% random deviations make held-out sequences differ beyond their starting phase. A uniformly random player wins about one third of rounds. Summary values and per-run traces are in `data/rps_results.json`.

The controls test whether a trained teacher's neural signal helps. A stronger claim of teaching requires a repeatable improvement over all three main controls on held-out performance or learning speed. Any observed difference across only three seeds is preliminary.

## Measured result

The trained teacher reached **60.6%** mean win rate on held-out sequences across three seeds (individual seed means 58.3%, 59.7%, 63.9%). It learned above the 33.3% random-play expectation, but its policy remained imperfect.

| Student condition | Last 3 training episodes, mean ± SD | Held out, disconnected, mean ± SD | Rounds to 60% threshold by seed |
|---|---:|---:|---|
| Alone | 62.0% ± 2.4 | 61.1% ± 1.1 | 168, 168, 216 |
| Trained teacher signal | 61.1% ± 3.0 | 61.1% ± 1.1 | 120, not reached, 240 |
| Untrained teacher signal | 61.6% ± 1.7 | 61.6% ± 1.3 | 288, 168, 216 |
| Shuffled trained signal | 60.2% ± 1.3 | 61.6% ± 1.3 | 144, 168, 240 |
| Shuffled wiring, alone | 35.2% ± 1.3 | 33.8% ± 1.3 | not reached in any seed |

The trained-signal and alone groups had **identical mean held-out win rates**. Their paired differences by seed were 0.0, +1.4, and −1.4 percentage points. The trained-signal group reached the threshold in two of three seeds, versus three of three alone. **This experiment provides no evidence that the teacher helped the student learn.** The shuffled-wiring control stayed near random play. These are exploratory results from three seeds, not a precise estimate of a small effect.

## Follow-up diagnostics

The [follow-up report](rps-diagnostics/follow-up-report.md) records a
bounded communication-channel calibration, a diagnosis of the RPS performance
plateau, and an isolated strategy-switch teaching test. A known two-label
signal was learned at **100%** held-out accuracy across 10 untouched
confirmation seeds with eight VPNs per label, 300 Hz stimulation, and a 120 ms
pulse; shuffled signals scored **50.4%**. This establishes that Fly B can
learn a synthetic channel code. It does not establish that Fly A sends useful
teacher-specific information. In the three-seed switch experiment, trained,
untrained, and shuffled teacher streams all aided early adaptation, with no
repeatable trained-teacher advantage on frozen held-out play.

The original browser demo and its result files still use the original channel.
The follow-up tests are isolated in separate scripts. The final teaching test
uses seeds 21–23, separate from channel calibration and earlier exploratory
RPS runs. Reproduce the tests in order:

```bash
node tools/calibrate_rps_channel.mjs screen
node tools/calibrate_rps_channel.mjs evaluate
node tools/calibrate_rps_channel.mjs evaluate eight_120ms_300hz
node tools/calibrate_rps_channel.mjs confirm
node tools/calibrate_rps_channel.mjs target_controls
node tools/audit_rps_channel_sources.mjs
python tools/analyze_rps_channel_normalization.py
node tools/diagnose_rps_plateau.mjs
node tools/probe_rps_action_updates.mjs
node tools/test_rps_teaching_switch.mjs final
node tools/test_rps_teaching_switch.mjs postonly_final
python tools/analyze_rps_teacher_signal.py final
python tools/plot_rps_channel.py
python tools/plot_rps_plateau.py
python tools/plot_rps_teaching.py final
```

The reports for each stage are [channel calibration](rps-diagnostics/channel-calibration.md),
[RPS plateau](rps-diagnostics/rps-plateau.md), and
[strategy-switch teaching](rps-diagnostics/teaching-switch-final.md).

