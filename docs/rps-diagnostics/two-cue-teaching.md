# Two-cue teaching with a locked Fly A source

The task stayed fixed: cue 0/1 stimulated VPN lanes 0/2 after the first eight neurons at 150 Hz; rewarded actions were 0/2. Fly B trained for ten 24-round episodes on mapping 0, then three episodes on mapping 1. Fly A learned mapping 1 separately for ten episodes before Fly B's switch. The existing network, connectome, action selection, and dopamine plasticity were unchanged. The first eight VPNs of lanes 0/1 carried the previously calibrated 300 Hz message from 120 to 240 ms. Source sampling at 120 ms occurred before Fly A's 240 ms choice; Fly A received no reward or learning updates while generating messages. All four students started from one shared mapping-0 training state per seed. Frozen testing used three new, teacher-disconnected episodes without learning.

## Locked source

On calibration seeds **11 and 12**, we trained separate Fly As on each mapping and compared them with untrained Fly A under identical cues and no evaluation feedback. We screened 42 MBON traces at 120 ms with thresholds 0.05, 0.2, 0.5, and 1. A candidate had to carry >0.03 bits about the mapping/correct action *given the cue* on each seed, differ from untrained activity on mapping 1, and have 20–80% active bits there. The highest mean conditional information among eligible candidates was **MBON 2132, trace ≥ 0.5**: 0.082 and 0.144 bits on the two seeds. Its active fraction after mapping-1 training was 75.0% and 70.8%, versus 4.2% and 1.4% untrained. The [source lock](../../data/two_cue_teacher_source_lock.json) was saved before the teaching runs. These are calibration associations across two learned mappings; within a fixed mapping, the cue already determines the answer.

The seed-11 pilot verified execution and was excluded from the preliminary comparison. On independent seeds **101–103**, Fly A scored 88.9%, 94.4%, and 91.7% while producing its frozen mapping-1 signal. Its bit was active on 55, 52, and 54 of 72 rounds; the untrained source was active on 0, 1, and 1 rounds. The source therefore changed with learning in these runs, but on cue 1 it was active on **every** round. The student had never been taught what either message bit meant during the old mapping.

## Fly B result

Accuracy means over three seeds; each frozen score averages three new 24-round games per seed. Round-level outcomes and individual seeds are in the raw JSON files.

| Condition | First 12 | First 24 | First 48 | Frozen, disconnected |
|---|---:|---:|---:|---:|
| No teacher | 11.1% | 37.5% | 57.6% | 67.6% |
| Trained Fly A | 11.1% | 13.9% | 22.2% | 24.1% |
| Untrained Fly A | 5.6% | 18.1% | 40.3% | 82.9% |
| Shuffled trained signal, within cue | 19.4% | 34.7% | 46.5% | 26.9% |

Trained signals lost to **every** control by round 24 and round 48, and did not leave a frozen benefit. All three seeds showed lower frozen accuracy with trained signals than with no teacher. A larger confirmation was not run.

The channel did reach Fly B. In a paired, frozen 24-round probe from the same student weights, changing only the message bit changed actions on **28/72** trials across the three seeds, changed the KC active set by about **48 neurons per round**, and changed the three MBON rates by a mean Euclidean distance of **1.35**. In natural first-48-round trials, KC activity rose from about 2 active KCs per cue without a teacher to about 25–38 with trained messages. For cue 1, the correct action 0 was chosen on **1/72** trained-message rounds, versus **55/72** no-teacher rounds. The trained bit stimulated lane 0 on every cue-1 round, adding a strong input that competes with the cue-1 lane-2 pattern. Shuffled trained messages retained the same per-cue bit frequencies and also gave poor frozen results. These observations support **cue interference from the communication stimulation** as the immediate failure. The lack of prior message interpretation training is an additional plausible limit, not a measured explanation of the performance gap.

## Next small experiment

Before another teaching run, make a frozen cue-preservation diagnostic: present the locked natural bit alongside each cue, compare cue-specific KC/MBON/action patterns against no message, and test **one weaker or shorter channel setting** against both cue preservation and the existing artificial-bit decoding control. Choose the setting without looking at a teaching outcome. Keep the source neuron, task, connectome, and learning rule fixed. This recommendation has not been implemented.

## Reproduce

The runner refuses to overwrite raw output. In a fresh copy without these generated files:

```powershell
foreach ($s in 11,12) { node tools/two_cue_transfer.mjs source $s 02 lanes02 }
node tools/two_cue_transfer.mjs teach 11 02 lanes02 # pilot, excluded
foreach ($s in 101,102,103) { node tools/two_cue_transfer.mjs teach $s 02 lanes02 }
python -m unittest flyhero.test_rps -v
```

Raw calibration: [seed 11](../../data/two_cue_source_02_11_lanes02.json), [seed 12](../../data/two_cue_source_02_12_lanes02.json). Raw pilot and comparisons: [11](../../data/two_cue_teach_02_11_lanes02.json), [101](../../data/two_cue_teach_02_101_lanes02.json), [102](../../data/two_cue_teach_02_102_lanes02.json), [103](../../data/two_cue_teach_02_103_lanes02.json). Only [the two-cue runner](../../tools/two_cue_transfer.mjs), this report, and the source lock were added or changed as code/report files for this experiment.
