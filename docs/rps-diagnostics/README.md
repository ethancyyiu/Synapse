# RPS teaching diagnosis

**Follow-up completed:** [calibrated channel, plateau diagnosis, and
strategy-switch teaching results](follow-up-report.md). The recommendation at
the end of this earlier exploratory report is historical; the linked
follow-up reports record what happened when it was tested.

This is a diagnostic study of the existing RPS simulation. No connectome edges,
neuron equations, dopamine learning rule, or browser task were changed. The
scripts replay the published three-seed experiment, run additional frozen
evaluations, and screen alternative opponents. All results below are exploratory
because there are only three seeds.

## Reproduce

From the repository root, with Node.js, Python, NumPy, and Matplotlib available:

```text
node tools/rps_signal_information.mjs
node tools/diagnose_rps.mjs signal
node tools/diagnose_rps.mjs current
node tools/diagnose_rps.mjs positive
node tools/diagnose_rps.mjs probe
node tools/diagnose_rps.mjs oracle_observed
node tools/diagnose_rps.mjs hard
node tools/diagnose_rps.mjs teacher long_sequence
node tools/diagnose_rps.mjs teacher probabilistic
node tools/diagnose_rps.mjs teacher switch
python tools/plot_rps_diagnostics.py
python -m unittest flyhero.test_rps -q
```

The JSON files in `data/rps_diagnostic_*.json` and
`data/rps_signal_information.json` contain the measurements and random seeds.
The plots are [round learning](round_learning.png),
[harder tasks](harder_tasks.png), [masked oracle control](oracle_control.png),
and [oracle with normal input](oracle_with_observation.png). Each plotted point
is a trailing **24-round** win fraction at that individual round, averaged
across seeds; shaded bands span the three seeds. The first plotted point is
round 24, when a full window is available. Thin vertical lines mark episodes.

## What an episode means

`RpsEpisode` runs **24 RPS rounds**. Each round advances the neuron simulation
for 400 one-millisecond steps, so an episode is 9,600 simulated milliseconds.
The episode resets voltages, spike counts, and traces at the start, but learned
KC→MBON synaptic multipliers persist. The opponent is reinitialized with a new
seed at each episode. The existing experiment trains for 12 episodes (288
rounds) per seed; the diagnostic alone and harder-task runs extend to 20
episodes (480 rounds).

## Does Fly B learn the current task?

Yes, in the limited sense that its **learned synaptic multipliers** improve
performance on new random sequences of the same opponent rule. At each
checkpoint, a fresh evaluator receives only Fly B's multipliers. It plays
three fixed 24-round held-out sequences with the channel disconnected and
learning frozen. The evaluator's random initialization and opponent sequences
are the same at every checkpoint, so the copied multipliers are the intended
change.

| Episodes trained | Rounds trained | Frozen held-out win rate, three-seed mean |
|---:|---:|---:|
| 0 | 0 | 17.6% |
| 1 | 24 | 19.4% |
| 2 | 48 | 28.2% |
| 3 | 72 | 38.9% |
| 5 | 120 | 54.2% |
| 10 | 240 | 61.1% |
| 20 | 480 | 60.6% |

The three seed-specific held-out rates after five episodes were 61.1%, 54.2%,
and 47.2%. The first training episode averaged 9.7% wins, episode five 54.2%,
and episode ten 65.3%; single-episode values are noisy. For a round-based
stability measure, we required a trailing 72-round win rate of at least 50%
and that threshold to hold for the next 72 windows. The first qualifying
rounds were 122, 105, and 193. Five episodes is already 120 rounds, and Fly B
has not mastered the task at that point.

The opponent's rule is simple: usually play the previous opponent move plus
one, with a 10% chance of a random replacement. Across the original training
sequences, that one-step predictor got **779/828 (94.1%)** of moves right when
a previous move existed. On the original held-out sequences, the corresponding
RPS policy won **92.1%** overall, including first rounds without a prior move.
Fly B's ~61% plateau is therefore far below the rule's achievable performance.
The rule's predictability helps explain the rapid early rise, while the frozen
checkpoint test shows that Fly B's weight changes matter.

## Early teacher comparison

The first five episodes of all four conditions were replayed from their saved
signals. Every replayed episode win rate matched `data/rps_results.json`.

| Student condition | Mean first 3 episodes | Mean first 5 episodes | Original disconnected held-out after 12 episodes |
|---|---:|---:|---:|
| Alone | 22.2% | 32.8% | 61.1% |
| Trained teacher signal | 31.5% | 39.7% | 61.1% |
| Untrained teacher signal | 27.8% | 35.3% | 61.6% |
| Shuffled trained signal | 27.3% | 37.8% | 61.6% |

The trained signal raises early training wins relative to alone, but shuffled
and untrained signals also raise them. With three seeds, this is not a
teacher-specific learning benefit, and the original final disconnected result
shows no trained-signal advantage.

## Signal delivery and information

In 72 paired first-episode rounds (24 per seed), the channel logged all 72
messages. Nonzero messages appeared in 48 rounds. The six target neurons
averaged roughly 29–34 spikes per round with trained signals, compared with
6–7 without them. MBON rates also changed. The signal changed 8, 3, and 6
actions in the three seeds, and the final plastic multipliers differed from
their matched alone runs. This confirms a neural and synaptic effect of
delivery, not a useful teaching effect.

For the saved 12-episode messages, 81.8% of trained-teacher vectors were
`(0,0,0)` or `(2,2,2)` after excluding the first round of each episode.
Their raw mutual information with the next move was 0.733 bits, but most of
that reflects the **previous opponent move already shown to Fly B**. After
conditioning on that previous move, empirical information was 0.022 bits per
round, near the 0.023-bit median from 200 within-observation permutations
(permutation p≈0.61). Untrained and shuffled signals likewise showed no
clear conditional advantage. This small sample does not prove zero
information; it gives no evidence of extra next-move prediction here. A
teacher could still communicate a learned rule or best-action mapping that
is redundant *statistically* with the previous move but useful to a novice
student. The performance controls and oracle probes test that possibility.

The signal is sampled at 180 ms, before the teacher acts or the current
opponent move is generated at 240 ms. The RPS learner encodes only
`previousOpponent`; it does not receive the rule, opponent seed, current move,
or previous player move. Its action rule is a fixed MBON argmax, not a
hardcoded RPS counter strategy. The original final evaluation freezes
learning and removes the channel, but tests only new sequences of the **same
opponent rule**, not a new rule. It has 72 held-out rounds per seed, so its
small mean differences are imprecise.

## Can Fly B interpret a deliberately informative signal?

The oracle positive controls are **separate from teacher communication**. An
artificial sender reveals the actual winning action, including stochastic
deviations, as a one-hot channel vector. A real teacher cannot know this
future random outcome.

When the normal previous-move input was masked, the oracle-trained student
reached **32.9%** frozen held-out after 20 episodes, versus **32.4%** with no
signal. In a 72-round frozen probe, each one-hot value drove about 13–14
spikes in its designated target pair, yet MBON action group 2 dominated for
all three values and the chosen action matched the oracle 24/72 times.

With normal previous-move input retained, oracle signals increased the first
five training episodes from **32.8% to 43.6%**, but held-out performance after
20 episodes was **59.3%** with the oracle signal versus **60.6%** alone. The
oracle remained available during that held-out control; it was disconnected
for the ordinary alone checkpoint. These results show a transient benefit,
not learned decoding of the oracle's action label.

## Harder opponents, unchanged circuit

Each screen used three seeds, 20 episodes of 24 rounds, the existing one-move
sensory encoding, and frozen held-out tests. Unless noted, random deviations
replaced the planned move 10% of the time. “Stable round” means a trailing
72-round win rate ≥50% sustained for another 72 windows. A seed can briefly
exceed 50% without generalizing.

| Opponent | Fly B first 5 episodes | Fly B held-out after 20 | Stable rounds by seed | Fly A held-out after 20 |
|---|---:|---:|---|---:|
| 12-move repeating sequence | 30.8% | 35.6% | none | 37.0% |
| Depends on previous 2 moves | 31.4% | 29.2% | none, 137, none | not run |
| Depends on previous 3 moves | 35.0% | 28.7% | none | not run |
| Switches offset +1 to +2 at episode 11 | 32.8% | 60.6% | 122, 105, 193 **before switch** | 62.5% |
| Previous-move rule, 45% random replacement | 22.5% | 40.7% | none | 46.3% |

The 12-move sequence was `[0,1,0,2,1,2,0,2,2,1,0,1]`, with a seed-specific
phase drawn from all 12 positions. The two history rules used
`(last1 + 2×last2 + 1) mod 3` and
`(last1 + last2 + 2×last3 + 1) mod 3`; the actual noisy move enters the next
history state. The switch occurred between rounds 240 and 241. Fly B's
first three post-switch episodes averaged 33–42% across seeds, then its
72-round rate recovered and stayed ≥50% through round 480 from rounds
428, 436, and 407. Fly A also recovered: its final three training episodes
averaged 63.4% and its frozen held-out mean was 62.5%.

The longer and history-dependent rules may require more than the single
previous move currently encoded for the learner. The unstable training peaks
on history2/history3 do not survive held-out evaluation. Increased randomness
lowers the achievable score as well as slowing learning, and the probabilistic
teacher did not reach the 50% stability criterion in two of three seeds.

## Recommendation

Keep the current task and neural architecture while fixing the **measurement
and channel validation** problem. First, calibrate the existing activity
channel on separate data so a deliberately informative one-hot signal can be
decoded at the MBON readout and improve a frozen held-out control. This can
change which existing sensory neurons receive the message or its presentation
timing, without changing the connectome or reward rule. Then choose teacher
source cells and quantization using a separate calibration set, requiring
unsaturated messages that reliably encode the teacher's action or learned
rule. Measure next-move information beyond the student's observation as a
separate check. Freeze those choices before comparing student outcomes; do
not select them on the final test seeds.

After those controls pass, the **strategy-switch adaptation task** is the
smallest plausible harder follow-up. It uses the existing one-move observation
and the same learned synapses, and both flies demonstrably adapt. Compare
recovery by individual post-switch round for alone, trained, untrained, and
shuffled signals, then evaluate with the channel disconnected. Use more than
three seeds for a confirmatory comparison. The current data do **not** show
that this switch, or any other tested harder rule, makes the present teacher
signals useful; that is a hypothesis for the next experiment.
