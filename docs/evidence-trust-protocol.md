# Study 3: can Fly B reject an outdated teacher? (evidence-counted trust)

Locked before any run of this study. New and separate: the trust-gate pilot and cohort, the stale-teacher studies, and every Science/banking artifact stay unchanged.

## What failed before, and why

The locked trust-gate cohort (`trust-gate-cohort-protocol.md`, 14 completed seeds) passed two of its three tests: it protected Fly B from the outdated teacher compared with unfiltered advice (+34.2 wins at 96 rounds, 95% CI +25.6 to +42.0) and kept the current teacher's benefit (-0.8 at 48 rounds, CI -1.6 to -0.1). It **failed no-harm**: gated students still ended 7.5 wins below learning alone at 96 rounds (CI -12.9 to -2.2).

The saved data point to the mechanism. Trust started at **1.0** and moved by 20% of the error, only on rounds where Fly B had played the advised move. Mean trust in the outdated teacher was already 0.017 by round 24, yet the loss against learning alone was bimodal across seeds: 0 to -1 wins on half of them, -23 to -24 on the rest. So the harm was done **early, while trust was still high**: full-strength imitation of the old rule deepened the old habit before enough evidence arrived, and on some seeds that habit outlasted the drop in trust.

## Hypothesis

A trust estimate that **starts neutral and weighs every observation equally**, trust = (rewarded matches + 1) / (all matches + 2) (a Beta(1,1) estimate of how often following the advice paid off), removes the harm of an outdated teacher while keeping the benefit of a current one. It starts at 0.5, not 1.0; after three losing matches it is 0.2, where the old rule was still at 0.51.

Comparison F asks whether *estimating* trust matters at all, or whether simply following advice half as hard does the same.

## Design

Copy of the trust-gate runner (`tools/check_trust_gate_pilot.mjs`): same RPS task, Strategy A history, rule change to Strategy B, both teachers trained by the same procedure and each gated at 58/72 on its own rule, 96 switch rounds, own-action rewards every round, measured DN spike decoding, and teacher-disconnected 72-game frozen checks at 0, 24, 48 and 96 rounds. Runner: `tools/check_evidence_trust.mjs`. The new students are defined in the runner; shared code is unchanged. Their only difference from the existing learner is the imitation strength (base rate 0.03, as before):

- **A** `alone`: no teacher.
- **B** `trained`: current teacher, unfiltered (strength 0.03).
- **C** `stale`: outdated teacher, unfiltered.
- **D** `evidenceStale`: outdated teacher, strength 0.03 x evidence trust.
- **E** `evidenceTrained`: current teacher, strength 0.03 x evidence trust.
- **F** `discountStale` and `discountTrained`: strength 0.03 x 0.5 always, no trust estimate.

Trust uses only Fly B's own action, the decoded advice, and its own reward on that round. It never sees which rule is active, which teacher is current, or whether the advice is correct.

## Seeds

The trust-gate formulas (teacher `28001 + 2s`, student `28002 + 2s`, and the same opponent formulas) at new indices. **Implementation check:** `s = 29` (code and runtime only; excluded). **Final:** `s = 30-44`. No study has used these indices (the trust-gate studies used 0-17). No seed is replaced; a seed whose current or outdated teacher fails its gate is reported as a teacher-readiness failure.

## Fixed analysis

`tools/analyze_evidence_trust.py`, written before any final run. Paired differences in frozen wins out of 72 over completed seeds, each with a 95% percentile bootstrap (10,000 resamples, fixed seed `2026111530`).

**Requirement 1, harm removed.** Both must hold:
- 1a. `evidenceStale - stale` at 96 rounds: lower bound > 4 (protection compared with unfiltered advice).
- 1b. `evidenceStale - alone` at 96 rounds: lower bound > -4 (no meaningful harm compared with learning alone).

Requirement 1 deliberately includes 1b. The earlier trust gate already passed 1a; only a method that also passes 1b improves on it.

**Requirement 2, benefit kept.** `evidenceTrained - trained` at 48 rounds: lower bound > -4.

**Result:** FULL PASS if at least 10 seeds complete and requirements 1 and 2 both hold; PARTIAL if exactly one holds; FAIL if neither.

**Reported, not tested:** every arm's mean at every checkpoint; `evidenceStale - discountStale` at 96 and `evidenceTrained - discountTrained` at 48 (does estimating trust beat a plain discount?); `discountStale - alone` at 96 and `discountTrained - trained` at 48; mean trust per checkpoint; per-seed differences, to show whether harm is still bimodal.

Nothing changes after final results are seen.

## Limits

One task with one rule change, and an outdated teacher whose advice Fly B can test on almost every round. Weighing advice by how well it has paid off is an established idea in action advising; this study tests whether a neutral, count-based start fixes the specific early-damage failure seen here, not a new principle.
