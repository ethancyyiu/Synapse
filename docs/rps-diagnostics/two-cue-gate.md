# Two-cue teaching gate

The first two-cue experiment **did not pass the task-readiness check**, so no source neuron was selected and no teaching comparison was run. These are one-seed diagnostics (seed 11), not a multi-seed teaching result. The connectome, neural dynamics, and dopamine learning rule were unchanged. **The follow-up below identifies a cue target pattern that passes a three-seed readiness check.**

## Task and measurements

Each 24-round episode has 12 presentations of each cue in shuffled order. Cue 0 or 1 stimulates the corresponding VPN lane after its first eight neurons; those eight remain reserved for the previously calibrated teacher channel. The existing `RpsBioRL` chooses among its three MBON actions. Only two actions are rewarded; an action outside the pair receives the ordinary incorrect-action punishment. The correct action is `actions[cue XOR mapping]`. Training uses +1 for correct and -1 for incorrect at the existing 240 ms decision and 340 ms dopamine timing. Each fly trained independently for ten episodes on mapping 0, then ten on mapping 1. Frozen evaluation used three new episodes per mapping. A constant policy choosing either rewarded action scores **50%** because cues are balanced.

| Rewarded actions | Fly | Mapping 0, last 3 episodes | Mapping 1 after reversal, last 3 | Frozen mapping 1 |
|---|---|---:|---:|---:|
| 0 and 1 | A | 43.1% | 11.1% | 6.9% |
| 0 and 1 | B | 43.1% | 5.6% | 2.8% |
| 0 and 2 | A | 84.7% | 38.9% | 44.4% |
| 0 and 2 | B | 90.3% | 38.9% | 48.6% |

With actions 0 and 2, Fly A first reached 75% in episode 3 and 95.8% in episode 5; Fly B first reached 79.2% in episode 4 and 91.7% in episode 5. This *initial* mapping is learnable, though untrained Fly A already scored 58.3% on it. Untrained Fly A scored only 20.8% on the reverse mapping. Training Fly A **from scratch** on the reverse mapping, rather than reversing its prior learning, produced 50.0% over its last three episodes and 48.6% frozen held-out accuracy. Its final 72 training choices were action 2 on 69 rounds, nearly a constant policy. On cue 1, the output rates for MBON groups 0 and 1 were zero in the first and last training windows, while group 2 remained active. Action 0, which is correct for cue 1 under the reverse mapping, was chosen on only 1 of the final 36 cue-1 rounds. This prevents reliable reinforcement of the needed cue-1/action-0 choice through the existing selected-action plasticity.

The 0/1 pair fails because action 1 is almost never selected. The 0/2 pair removes that particular problem, but the model favors the original cue-to-action association so strongly that it does not learn the reverse mapping in this short test. Reversal therefore has room for improvement but is **not yet a suitable teaching test**: a pretrained Fly A does not know the new mapping, and Fly B's apparent early progress on mapping 0 is partly an initial action bias. A source scan or three-seed student comparison would measure that task failure rather than knowledge transfer.

## Recommendation and cleanup

Keep the two-cue design and, in a separate next step, identify cue stimulation patterns that give both rewarded actions a measurable chance to win for **both** mappings. Test each mapping from scratch and after reversal before selecting teacher neurons. If both mappings become learnable, use calibration-only source selection and the locked VPN communication channel, then compare trained, untrained, shuffled, and no-teacher conditions on three independent seeds. More cues are not indicated by these measurements: Fly B did not master the reversal quickly.

Git inspection found pre-existing uncommitted changes in `README.md`, `docs/lab-notebook.md`, `index.html`, and `js/brain.js`; they were left untouched. The older RPS scripts and raw outputs are referenced by diagnostic reports or retain reproduction value. No file was verified safe to delete. Older trace and plotting helpers are uncertain cleanup candidates and remain in place. This work added only the isolated runner, this report, and the four raw JSON files below.

## Reproduction and raw data

Run from a fresh copy without the generated `two_cue_*.json` files (the runner refuses to overwrite measurements):

```powershell
node tools/two_cue_transfer.mjs baseline 11 01
node tools/two_cue_transfer.mjs baseline 11 02
node tools/two_cue_transfer.mjs fresh1 11 02
node tools/two_cue_transfer.mjs probe 11 02
python -m unittest flyhero.test_rps -v
python -m flyhero.test
```

The runner is [two_cue_transfer.mjs](../../tools/two_cue_transfer.mjs). Raw, round-level measurements: [0/1 reversal](../../data/two_cue_baseline_01_11.json), [0/2 reversal](../../data/two_cue_baseline_02_11.json), [0/2 reverse from scratch](../../data/two_cue_fresh1_02_11.json), and [untrained probe](../../data/two_cue_probe_02_11.json). There are no teacher signals or teaching outcome measurements for this task yet.

Validation: all six RPS tests passed; the original Fly Hero neuron, wiring, reward, browser/Python parity, and Bio-RL learning checks passed. The new runner passed `node --check`, and every training episode in the saved data has exactly 12 rounds per cue.

## Follow-up: action bias and limited cue calibration

All follow-up numbers use rewarded actions 0 and 2. Diagnostic logging reproduced the original seed-11 reverse-mapping action/reward sequence exactly. In the **original VPN lanes 0/1** at 150 Hz, the last three episodes had the following counts (36 rounds per cue):

| Mapping | Cue | Actions 0/1/2 | Correct rewards by action 0/1/2 | Mean active KCs | Mean MBON rates 0/1/2 |
|---|---:|---:|---:|---:|---|
| Initial | 0 | 28/0/8 | 28/0/0 | 1.6 | 0.438/0.117/0.224 |
| Initial | 1 | 2/1/33 | 0/0/33 | 0.8 | 0/0/0.313 |
| Reverse, from scratch | 0 | 1/0/35 | 0/0/35 | 1.7 | 0.115/0.069/0.298 |
| Reverse, from scratch | 1 | 1/1/34 | 1/0/0 | 0.6 | 0/0/0.335 |

The cue KC sets are distinct: in the failed reverse run, mean KC-set Jaccard overlap was 0.557 within a cue and 0.000 between cues. The issue is sparse cue-1 recruitment and almost no MBON-0 output. Across all 120 cue-1 rounds of that run, the correct action 0 was selected and rewarded **9 times**. Its summed absolute KC→MBON weight change on those rounds was 2.82. Thus it had few chances for selected-action learning.

Forced balanced exploration was confined to training: each cue received 60 action-0 and 60 action-2 choices over ten episodes, with unchanged dopamine and synaptic updates. Cue-1/action-0 rewards rose to 60 and summed absolute weight change to 11.79, yet **normal, frozen** action selection scored 47.2%, compared with 48.6% after ordinary training. Exploration alone did not solve this cue pattern.

Two limited cue changes were checked, still reserving the first eight VPN neurons of every lane for communication. Raising lanes 0/1 from 150 to 300 Hz gave 98.6% frozen accuracy on the initial mapping but only **4.2%** on the reverse. Presenting cue 1 on **VPN lane 2 instead of lane 1**, at the original 150 Hz, worked on seed 11: reverse-mapping cue 1 recruited 2.8 active KCs per round, MBON-0 rate averaged 1.345, and action 0 was selected on 34/36 late cue-1 rounds. Across its 120 cue-1 training rounds, action 0 was rewarded 78 times with summed absolute weight change 32.18. The first-eight VPN communication targets remain free.

The lane-0/2 pattern was then checked on **seeds 11, 12, and 13**, without further tuning. Frozen held-out accuracy after training each mapping *from scratch* was 90.3/91.7/97.2% for mapping 0 and 94.4/87.5/94.4% for mapping 1. After ten old-mapping episodes followed by ten reverse-mapping episodes, frozen reverse accuracy was **88.9/93.1/90.3% for Fly A** and **91.7/95.8/95.8% for Fly B**. Fly B's first post-switch episode scored 45.8/37.5/33.3%; its second scored 79.2/70.8/70.8%. Both mappings therefore pass the preliminary readiness gate, but independent Fly B recovery is fast. A later teaching experiment should focus on the first 24–48 post-switch rounds and retain no-teacher, untrained, and shuffled controls. **No teacher signals or teaching outcomes were tested here.**

Raw follow-up measurements are the `data/two_cue_diag0_02_*`, `data/two_cue_diag1_02_*`, `data/two_cue_explore1_02_11.json`, `data/two_cue_probe_02_11_lanes02.json`, `data/two_cue_probe_02_11_300hz.json`, and `data/two_cue_baseline_02_*_lanes02.json` files. To reproduce in a fresh copy without those output files:

```powershell
node tools/two_cue_transfer.mjs diag0 11 02
node tools/two_cue_transfer.mjs diag1 11 02
node tools/two_cue_transfer.mjs explore1 11 02
node tools/two_cue_transfer.mjs probe 11 02 lanes02
node tools/two_cue_transfer.mjs probe 11 02 lanes01 300
node tools/two_cue_transfer.mjs diag0 11 02 lanes01 300
node tools/two_cue_transfer.mjs diag1 11 02 lanes01 300
foreach ($s in 11,12,13) {
  node tools/two_cue_transfer.mjs diag0 $s 02 lanes02
  node tools/two_cue_transfer.mjs diag1 $s 02 lanes02
  node tools/two_cue_transfer.mjs baseline $s 02 lanes02
}
```
