# Advice budget follow-up: 15-seed result

This is a synthetic RPS rule-switch simulation, not biological or banking evidence. The [protocol](advice-budget-followup-protocol.md) was fixed before the final 15-seed run. Pilot seeds 98 and 99 are excluded. The earlier [12-message result](advice-budget-research.md) and every existing Science/banking artifact were preserved. These final seeds reuse seeds from that earlier study, so this is a controlled follow-up, not untouched independent confirmation.

## Question 2: How much advice?

All 15 trained teachers passed their 58/72 gate. Every scheduled preference was decoded from measured DN spikes. Every learner had 48 training rounds, its own rewards, and a teacher-disconnected 72-game frozen evaluation. The runner reproduced the archived no-advice, trained-12, random-12, and trained-46 scores exactly for every seed.

| Trained messages / 46 opportunities | Mean wins / 72 | Students >=58/72 | Random mean | Shuffled mean | Paired gain vs no advice, 95% CI | Paired gain vs random, 95% CI | Paired gain vs stronger matched control, 95% CI | Preset in-sample recovery criterion |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 43.73 | 2/15 | 13.67 | 17.53 | +31.53 [22.80, 39.87] | +30.07 [21.60, 38.13] | +24.13 [17.40, 30.60] | FAIL |
| 24 | 55.60 | 4/15 | 22.07 | 16.40 | +43.40 [36.40, 50.40] | +33.53 [24.33, 43.00] | +29.00 [21.13, 37.80] | FAIL |
| 36 | 64.47 | 13/15 | 21.60 | 25.20 | +52.27 [45.27, 59.27] | +42.87 [30.40, 54.13] | +34.60 [23.13, 45.87] | PASS |
| 46 | 68.07 | 14/15 | 24.60 | 23.87 | +55.87 [50.13, 61.53] | +43.47 [33.40, 53.13] | +36.73 [27.73, 45.80] | PASS |

No-advice mean was 12.20/72. All differences are computed within the same seed; brackets are 95% paired percentile bootstrap intervals from 10,000 resamples. The stronger-control comparison takes the higher of the random and shuffled scores *within each seed*. The preset criterion required mean >=58/72, at least 12/15 individual scores >=58/72, and a lower paired interval >4 wins over that stronger control. **36 is the smallest of the four tested budgets that met this sample criterion.** The result does not identify the exact minimum; 25–35 messages were not tested, and a new independent cohort is needed to estimate general reliability. The 24-message mean fell below 58 and only 4/15 students recovered, despite clear improvement over controls.

### Individual frozen wins / 72

| Seed | None | 12 | 24 | 36 | 46 |
|---:|---:|---:|---:|---:|---:|
| 100 | 2 | 47 | 53 | 71 | 71 |
| 101 | 8 | 22 | 47 | 40 | 65 |
| 102 | 30 | 60 | 48 | 67 | 67 |
| 103 | 24 | 56 | 71 | 71 | 71 |
| 104 | 2 | 52 | 52 | 55 | 57 |
| 105 | 25 | 25 | 55 | 58 | 68 |
| 106 | 0 | 19 | 45 | 69 | 69 |
| 107 | 1 | 46 | 54 | 69 | 70 |
| 108 | 0 | 46 | 57 | 65 | 66 |
| 109 | 24 | 56 | 57 | 67 | 69 |
| 110 | 0 | 61 | 65 | 67 | 68 |
| 111 | 2 | 48 | 71 | 71 | 71 |
| 112 | 24 | 46 | 58 | 67 | 69 |
| 113 | 25 | 47 | 54 | 65 | 70 |
| 114 | 16 | 25 | 47 | 65 | 70 |

## Question 3: How often and can B request?

**The planned equal-budget timing/request comparison was not run.** Part 1's four-budget, matched-control neural simulation was already substantial, and the protocol explicitly allowed stopping after Part 1 if runtime became excessive. Consequently this follow-up cannot rank early, evenly spaced, random-time, and student-requested advice, or show that B requests fewer messages while retaining recovery.

Existing evidence remains limited: the earlier 12-message study found early and spaced means of 41.20 and 43.73 at round 48, a descriptive difference of 2.53 wins favoring spacing. A different, previously saved 288-round experience-count request pilot ([report](adaptive-sparse-teaching-report.md)) **failed its teacher-specific criterion**: adaptive trained advice did not sufficiently beat exposure-matched shuffled advice. That pilot cannot answer the new 48-round, score-margin request question. Asking for help from Fly B's own uncertainty remains an unverified hypothesis here.

## Judge explanation and limits

Fly A sends a short suggestion naming one of three actions. Fly B still acts and learns from its own reward; measured neural spikes deliver the suggestion, which adds a small imitation update. With this fixed simulator, 12 or 24 suggestions helped but did not recover reliably; 36 of 46 opportunities met the preset sample target. We do not know the exact minimum, the best timing, or whether a request policy can save messages. The spike transmitter and potential reverse request are engineered software channels. No result here proves biological fly teaching, real-world transfer, or an advantage of spikes over an information-matched digital message.

Saved [aggregate JSON](../data/advice_budget_followup_summary.json) and write-once `data/advice_budget_followup/seed_100.json` through `seed_114.json` contain actual counts and per-seed outcomes. Analysis: `tools/analyze_advice_budget_followup.py`. No old result file was overwritten.
