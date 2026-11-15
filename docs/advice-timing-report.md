# Study 2 result: timing made no detectable difference; student requests saved messages but did not recover

[Protocol](advice-timing-protocol.md), locked before the pilot and the final runs. Simulated Rock-Paper-Scissors only.

**Verdicts under the locked rules**
- **Timing (36 messages): no detectable difference** at round 48 between early, evenly spaced and randomly timed schedules.
- **Student requests: NOT SUPPORTED.** Fly B asked for far fewer messages (10.2 on average) and those messages clearly helped compared with random or shuffled advice at the same moments, but it did not recover and did much worse than 36 spaced messages.

## What was run

- Same task, teacher, student, learning rule, spike code and evaluation as the advice-budget follow-up; 48 rule-B training rounds.
- Request threshold `tau = 0.20`, kept by the pre-written pilot rule (pilot seed 299 made 12 requests; excluded).
- **15 fresh seeds (300-314).** All 15 teachers passed their gate (62-70 of 72). Every scheduled message decoded. No seed failed or was replaced.

## Results (frozen, teacher disconnected; wins out of 72)

| Arm | Messages | Mean at 24 | Mean at 48 | Students at or above 58 at 48 |
|---|---:|---:|---:|---:|
| No advice | 0 | 2.73 | 8.40 | 0/15 |
| Early 36 | 36 | **50.00** | 63.60 | 12/15 |
| Evenly spaced 36 | 36 | 39.87 | 62.67 | 11/15 |
| Random-timed 36 | 36 | 43.13 | 61.87 | 11/15 |
| **Student-requested** | **10.2 (6-18)** | 11.87 | **38.13** | **1/15** |
| Random advice at requested times | same | 3.07 | 16.53 | 0/15 |
| Shuffled advice at requested times | same | 5.93 | 17.73 | 0/15 |
| All 46 (reference) | 46 | 50.00 | 67.73 | 15/15 |

Paired differences at round 48 (95% percentile bootstrap, 10,000 resamples, seed 2026111560):

| Contrast | Mean | 95% CI |
|---|---:|---|
| Early minus spaced | +0.93 | -3.13 to +4.73 |
| Random-timed minus spaced | -0.80 | -4.73 to +2.67 |
| Requested minus spaced | **-24.53** | **-33.87 to -16.27** |
| Requested minus stronger yoked control | **+15.93** | **+10.00 to +21.67** |
| Requested minus no advice | +29.73 | +22.07 to +37.87 |
| All 46 minus spaced 36 | +5.07 | +2.20 to +8.33 |

Request criteria: fewer messages, yes; recovers, **no** (1/15, mean 38.13); not worse than spaced, **no**; content matters, yes.

### Per-seed scores at round 48

| Seed | Teacher | None | Early | Spaced | Random-timed | Requests | Requested | Req. random | Req. shuffled | All 46 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 65 | 27 | 62 | 62 | 65 | 11 | 52 | 48 | 29 | 65 |
| 301 | 68 | 25 | 62 | 64 | 57 | 12 | 49 | 1 | 49 | 66 |
| 302 | 70 | 0 | 69 | 69 | 69 | 16 | 47 | 24 | 0 | 69 |
| 303 | 65 | 1 | 70 | 70 | 59 | 7 | 24 | 1 | 1 | 69 |
| 304 | 67 | 1 | 62 | 63 | 67 | 9 | 46 | 1 | 24 | 68 |
| 305 | 62 | 3 | 57 | 53 | 56 | 6 | 24 | 24 | 24 | 63 |
| 306 | 65 | 12 | 70 | 70 | 70 | 13 | 28 | 1 | 1 | 70 |
| 307 | 65 | 0 | 58 | 51 | 58 | 11 | 46 | 46 | 22 | 60 |
| 308 | 66 | 0 | 50 | 51 | 52 | 8 | 23 | 0 | 0 | 67 |
| 309 | 65 | 24 | 69 | 50 | 60 | 9 | 47 | 24 | 24 | 70 |
| 310 | 69 | 2 | 49 | 69 | 49 | 6 | 2 | 2 | 2 | 71 |
| 311 | 66 | 1 | 67 | 65 | 68 | 11 | 46 | 23 | 31 | 68 |
| 312 | 66 | 1 | 70 | 62 | 66 | 7 | 24 | 1 | 1 | 70 |
| 313 | 66 | 25 | 70 | 70 | 63 | 9 | 45 | 25 | 25 | 71 |
| 314 | 65 | 4 | 69 | 71 | 69 | 18 | 69 | 27 | 33 | 69 |

## What this establishes, and what it does not

- **Timing within 36 messages did not matter by round 48** at this sample size: all three schedules ended within about 1 win of each other. Early advice was well ahead at round 24 (50.00 versus 39.87 for spaced), so sending advice early speeds the first half, but the other schedules caught up. This is a descriptive difference at 24, not a tested one.
- **Fly B's requests were genuinely informative but far too few.** The requested messages beat random and shuffled advice delivered at the same moments by about 16 wins, so they carried the teacher's knowledge. But Fly B stopped asking long before it had learned the new rule: it averaged 10 requests out of a possible 36, and its score stayed far below 36 spaced messages.
- **Why it under-asked (looked at only after the result, not part of the test).** The rule asks when Fly B was unsure *last time* it saw a cue. Right after the rule change, Fly B is often confidently wrong: its old habit still gives it a large score gap, so it does not ask, even though it is losing. Score-gap uncertainty measures confidence, not correctness. A request rule that also used recent losses might do better, but that would be a new, separately locked study.
- No optimal schedule is shown, and nothing here says how a request rule should be designed in general.

Files: per-seed `data/advice_timing/seed_300.json`-`seed_314.json` (each includes the full request log), pilot `pilot_seed_299_tau_0.2.json`, summary `data/advice_timing_summary.json`. Runner `tools/check_advice_timing.mjs`; analysis `tools/analyze_advice_timing.py`.
