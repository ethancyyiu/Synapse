# Study 3 result: evidence-counted trust removed the harm of an outdated teacher, but slowed learning from a current one

**Verdict under the locked rule: PARTIAL.** Requirement 1 (harm removed) held; requirement 2 (benefit kept) failed. [Protocol](evidence-trust-protocol.md), locked before any run. Simulated Rock-Paper-Scissors only.

## What was run

- Copy of the trust-gate runner: rule-A history, 96 rule-B rounds, own-action reward every round, measured DN spikes, frozen teacher-disconnected checks at 0, 24, 48, 96. Two teachers (current rule B, outdated rule A), each gated at 58/72 on its own rule.
- Two new students defined in the runner (shared code unchanged): **evidence trust**, imitation strength 0.03 x (rewarded matches + 1) / (all matches + 2), starting at 0.5; and **discount**, strength 0.03 x 0.5 always.
- **15 fresh seeds (30-44).** All 30 teachers passed their gates. Every arm decoded all 92 messages. No seed failed or was replaced. Implementation check seed 29 excluded.

## Results (frozen, teacher disconnected; mean wins out of 72)

| Arm | 24 | 48 | 96 |
|---|---:|---:|---:|
| No teacher | 2.1 | 11.3 | 25.7 |
| Current teacher, unfiltered | 54.5 | **67.3** | 69.7 |
| Outdated teacher, unfiltered | 0.7 | 0.7 | **0.8** |
| Outdated teacher, evidence trust | 0.7 | 8.4 | **24.4** |
| Current teacher, evidence trust | 34.1 | **60.0** | 69.7 |
| Outdated teacher, discount 0.5 | 0.7 | 0.7 | 0.8 |
| Current teacher, discount 0.5 | 30.2 | 55.6 | 69.7 |

Mean trust in the outdated teacher fell to 0.049 by round 24 (0.015 by 96); trust in the current teacher rose to 0.66, 0.89 and 0.96.

### Locked tests (95% percentile bootstrap, 10,000 resamples, seed 2026111530)

| Test | Mean | 95% CI | Needed | Holds |
|---|---:|---|---|---|
| 1a. Protection: evidence-outdated minus unfiltered-outdated, 96 | +23.60 | +16.07 to +30.93 | lower > 4 | yes |
| 1b. No harm: evidence-outdated minus no teacher, 96 | -1.27 | -2.73 to 0.00 | lower > -4 | **yes** |
| 2. Benefit kept: evidence-current minus unfiltered-current, 48 | -7.33 | -11.73 to -3.27 | lower > -4 | **no** |

Per-seed no-harm differences: 0 on 12 seeds, -4, -8 and -7 on three. Per-seed benefit-kept differences: 0 on five seeds, -2 to -5 on five, and -11, -13, -21, -22, -24 on five.

### Reported comparisons

| Contrast | Mean | 95% CI |
|---|---:|---|
| Evidence-outdated minus discount-outdated, 96 | +23.60 | +16.07 to +30.93 |
| Evidence-current minus discount-current, 48 | +4.40 | +1.73 to +7.73 |
| Discount-outdated minus no teacher, 96 | -24.87 | -32.93 to -16.87 |
| Discount-current minus unfiltered-current, 48 | -11.73 | -16.33 to -7.13 |
| Unfiltered-outdated minus no teacher, 96 | -24.87 | -32.93 to -16.87 |
| Unfiltered-current minus no teacher, 48 | +56.07 | +50.40 to +61.60 |

## What this establishes, and what it does not

- **The earlier failure was fixed.** The earlier trust gate (trust starting at 1.0) left students 7.5 wins below learning alone. Starting trust at a neutral 0.5 and counting every observation equally removed that harm: on 12 of 15 seeds the outdated-teacher student matched learning alone exactly.
- **But skepticism has a cost.** The same rule distrusts a *current* teacher at first, too, since Fly B often loses while its old habit is still strong, so it slowed learning: about 7 wins behind at round 48, with large delays on 5 seeds. By round 96 every current-teacher arm reached the same score. Both requirements were not met together, so this is a partial result: a trade-off, not a solution.
- **Estimating trust is what protects.** Simply following advice half as hard gave no protection at all (as harmed as unfiltered advice) and was slower with a good teacher. The protection comes from learning *whose* advice pays off.
- Trusting advice by its payoff is established in action advising (e.g. negative-transfer work such as Zhan et al. 2016). What this study adds is a measured trade-off for this simulation: where the trust estimate starts decides whether a bad teacher harms Fly B or a good teacher is slowed.

Files: per-seed `data/evidence_trust/seed_30.json`-`seed_44.json`, implementation check `smoke_seed_29.json`, summary `data/evidence_trust_summary.json`. Runner `tools/check_evidence_trust.mjs`; analysis `tools/analyze_evidence_trust.py`.
