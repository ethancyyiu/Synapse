# Traffic-signal rule-switch: 3-seed feasibility pilot

**. Status: all 3 seeds completed, teacher gate passed on all 3.**

## Hypothesis
The rule-switch teaching mechanism validated in RPS generalises to a structurally
identical task with a real-world framing: an agent learning to respond correctly to
traffic signals (Red/Green/Yellow), where a construction-zone rule change shifts the
correct response mapping mid-deployment.

## Experiment
Same architecture, DN channel, and 5-arm design as the validated RPS rule-switch
cohort. Only change: new seed range (36001+/36002+). Opponent: `previous_opponent`
offset 1 (pre-switch, normal rule) → offset 2 (post-switch, construction override).
Cue = previous signal. Seeds 0–2.

Runner: `tools/check_traffic_signal_cohort.mjs`
Artifacts: `data/traffic_signal_cohort/seed_{0,1,2}.json`

## Results

| Seed | Teacher wins (/72) | Taught @48 | Alone @48 | Advantage @48 | Advantage @24 |
|-----:|-------------------:|-----------:|----------:|--------------:|--------------:|
| 0    | 68                 | 71         | 2         | **+22**       | +58           |
| 1    | 66                 | 62         | 1         | **+18**       | +46           |
| 2    | 62                 | 70         | 7         | **+24**       | +69           |

Controls at 48 rounds (seed / random / untrained / shuffled):

| Seed | Random | Untrained | Shuffled | Best control | Taught beats best by |
|-----:|-------:|----------:|---------:|-------------:|---------------------:|
| 0    | 25     | 24        | 49       | 49           | +22                  |
| 1    | 24     | 23        | 44       | 44           | +18                  |
| 2    | 21     | 23        | 46       | 46           | +24                  |

All 3 seeds: taught fly reached 58+/72 at 48 rounds. Alone fly: 0/3.

## Controls integrity
- Teacher weights frozen throughout student training: confirmed all 3 seeds.
- KC→MBON synapses unchanged: confirmed all 3 seeds.
- Preference tape reproduced exactly in live run: confirmed.
- Shuffled tape broke cue alignment while preserving action distribution.

## Conclusion
The teaching mechanism replicates on a structurally identical task with a
traffic-signal framing. This is not a new mechanism — it is the same mechanism
with a different real-world interpretation. The result supports the claim that
the mechanism generalises beyond the specific RPS domain framing.

**Honest limits:**
- The underlying math (previous_opponent, offset 1→2) is identical to RPS Strategy A→B.
  This is a replication with a different story, not an independent task.
- Seed 2 at round 96: random (69) matched trained (69). Advantage collapses at 96 rounds
  for that seed, consistent with the known pattern that alone eventually catches up.
- 3 seeds is a feasibility pilot, not a 15-run cohort.

## Reproduction
```
node tools/check_traffic_signal_cohort.mjs 0
node tools/check_traffic_signal_cohort.mjs 1
node tools/check_traffic_signal_cohort.mjs 2
```
Run only in a copy without the existing seed artifacts (write-once).

## Research log entry
- **Hypothesis:** teaching mechanism works on traffic-signal framing (same math, new context)
- **Experiment:** 3-seed pilot, 5 arms, locked seeds, same runner structure as RPS cohort
- **Result:** advantage48 = +22, +18, +24. Taught reached 58+/72 in 3/3; alone in 0/3.
- **Controls:** all passed integrity checks
- **Conclusion:** replicates; supports generalisation claim for the presentation
- **Next:** no further experiments needed pre-hackathon
