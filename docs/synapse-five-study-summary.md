# Synapse: five focused studies

Five investigations, run in the order 5, 2, 3, 4, 1. Every protocol was written and locked before its final data existed. Every study used new seeds that no earlier study had used, and nothing was tuned after final results. Failures, partial results, teacher-readiness failures and one recording bug are all reported. Everything here is a simulation of Rock-Paper-Scissors-style tasks; none of it is biological or banking evidence. No earlier result, protocol, Science demo or banking file was changed.

## At a glance

| # | Question | Hypothesis | Prior evidence | New result | Verdict | Main limitation | Protocol / results |
|---|---|---|---|---|---|---|---|
| 5 | Does the 36-message result hold on fresh seeds? | 36 spaced messages give mean recovery and a teacher-specific benefit | 64.47/72, 13/15 recovered, on seeds reused from an earlier study | 60.20/72, **9/15** recovered; lead over matched controls +26.47 (CI +18.33 to +35.53) | **PARTIAL** | One task; exact minimum budget unknown | [protocol](advice-budget-replication-protocol.md), [report](advice-budget-replication-report.md), `data/advice_budget_replication_summary.json` |
| 2 | Does timing matter at 36 messages? Can Fly B ask for advice? | Timing changes the outcome; requests reach recovery with fewer messages | Early vs spaced 12 messages differed by 2.5 wins (descriptive); an earlier request pilot failed its content test | No timing difference at round 48 (early minus spaced +0.93, CI -3.13 to +4.73). Requests used 10.2 messages, carried real teacher content (+15.93 over yoked controls), but recovered only 1/15 | **Timing: no difference. Requests: NOT SUPPORTED** | Request rule uses previous same-cue confidence, not correctness | [protocol](advice-timing-protocol.md), [report](advice-timing-report.md), `data/advice_timing_summary.json` |
| 3 | Can Fly B reject an outdated teacher without losing a good one? | Neutral-start, evidence-counted trust removes harm and keeps benefit | Old trust gate protected but left -7.5 wins of harm | Harm removed: -1.27 vs learning alone (CI -2.73 to 0.00). Good teacher slowed: -7.33 at round 48 (CI -11.73 to -3.27); caught up by 96 | **PARTIAL** | One rule change; a trade-off, not a solution | [protocol](evidence-trust-protocol.md), [report](evidence-trust-report.md), `data/evidence_trust_summary.json` |
| 4 | Does teaching survive a second rule change? | Advice helps after a change back to an earlier rule | Every confirmed result had one change; noisy-opponent test failed | Change 1: +27.79 lead. Change 2 (back to the old rule): Fly B alone recovered within 24 rounds; lead -0.07 (CI -0.21 to 0.00). 1 teacher-readiness failure | **FAIL** | Only two non-trivial rules exist with this cue | [protocol](repeated-change-protocol.md), [report](repeated-change-report.md), `data/repeated_change_summary.json` |
| 1 | Does the connectome contribute beyond carrying the message? | Fly A's neurons carry its advice; real wiring helps teaching | Spikes and digital were identical in a 2-student ablation; disruption changed no actions | No natural teaching signal (stopped). Spikes and digital identical on 15/15 seeds. Rewired Fly A could not learn (22-25 of 72 on 15/15), so the wiring-and-teaching test stopped at teacher readiness | **E stopped; A=B; D stopped at readiness** | Input scaling fitted on real wiring | [protocol](connectome-contribution-2-protocol.md), [report](connectome-contribution-2-report.md), `data/endogenous_signal_summary.json`, `data/connectome_contribution_2_summary.json` |

## Study 5: fresh-seed replication of the 36-message result

**Method.** An exact copy of the earlier follow-up's runner, restricted to five arms (no advice, 36 trained, 36 random, 36 shuffled, 46 trained). Before use, it reproduced an archived seed's saved scores exactly. 15 new seeds (200-214).

**Seeds.** 15 attempted, 15 completed; every teacher passed (59-69 of 72); every message decoded.

**Results at round 48.** No advice 13.07; 36 random 25.47; 36 shuffled 25.60; **36 trained 60.20 (9/15 at 58 or above)**; 46 trained 65.73 (15/15). 36 trained minus the stronger matched control: +26.47 (95% CI +18.33 to +35.53).

**Deviations.** None.

**What it shows.** The benefit of 36 messages comes from the teacher's content, and that replicates. "36 messages reliably bring recovery" does not: over the two cohorts, 13/15 and then 9/15. 36 is borderline here; 46 recovered 15/15.

## Study 2: advice timing and student requests

**Method.** 36 messages delivered early, evenly spaced, or at random times. A request arm asked for advice when Fly B's score gap was small the last time it saw the same cue (threshold 0.20, kept by a pre-written pilot rule). Random and shuffled advice were delivered at exactly the requested moments. 15 new seeds (300-314).

**Seeds.** 15 attempted, 15 completed; every teacher passed (62-70); every message decoded. Pilot seed 299 excluded.

**Results at round 48.** Early 63.60, spaced 62.67, random-timed 61.87; requested 38.13 with 10.2 messages (1/15 recovered); random at requested times 16.53; shuffled at requested times 17.73; all 46 messages 67.73. Early minus spaced +0.93 (CI -3.13 to +4.73); random-timed minus spaced -0.80 (CI -4.73 to +2.67); requested minus spaced -24.53 (CI -33.87 to -16.27); requested minus the stronger yoked control +15.93 (CI +10.00 to +21.67).

**Deviations.** None (the pilot outcome was recorded in the protocol before the final seeds).

**What it shows.** Within 36 messages, *when* they are sent did not matter by round 48, although early advice was well ahead at round 24 (50.0 versus 39.9). Fly B's own requests were genuinely useful but far too few. After the change, Fly B is often confidently wrong, so a confidence-based rule does not ask when it most needs to. That last point is an interpretation after the fact, not a test. No optimal schedule is claimed.

## Study 3: rejecting an outdated teacher

**Method.** The earlier trust gate failed because trust started at 1.0, so Fly B fully imitated an outdated teacher before gathering evidence. The new rule: trust = (rewarded matches + 1) / (all matches + 2), starting at 0.5. Compared with unfiltered advice and with a plain "follow advice half as hard" baseline, for both a current and an outdated teacher. 96 rounds. 15 new seeds (30-44). "Harm removed" included the no-harm test the old method failed.

**Seeds.** 15 attempted, 15 completed; all 30 teachers passed; every message decoded.

**Results.** Protection (outdated teacher, trust versus unfiltered, round 96): +23.60 (CI +16.07 to +30.93). No harm (versus learning alone, round 96): -1.27 (CI -2.73 to 0.00), with 12 of 15 seeds at exactly 0. Benefit kept (current teacher, trust versus unfiltered, round 48): **-7.33 (CI -11.73 to -3.27)**; equal by round 96 (69.7 versus 69.7). The half-strength baseline gave no protection (-24.87 versus alone) and was slower with a good teacher.

**Deviations.** None.

**What it shows.** Where the trust estimate starts decides the trade-off. A neutral start removes the harm of a bad teacher but delays benefiting from a good one. Estimating trust from outcomes, not merely discounting advice, is what protects.

## Study 4: a second rule change

**Method.** Rule A history, then rule B (48 rounds), then back to rule A (48 rounds). A current teacher for each phase. Trained, random and shuffled advice against no advice. 15 new seeds (0-14, new formulas).

**Seeds.** 15 attempted, 14 completed. Seed 9 stopped at teacher readiness: its rule-B teacher scored 57 of 72, one short of the gate. It was not replaced.

**Results.** After change 1: trained 69.50 versus the best control (lead +27.79, CI +18.64 to +37.14). After change 2: no advice 69.43 already by round 72; at round 96 the lead was -0.07 (CI -0.21 to 0.00). At round 72 the taught students were 3.86 behind (CI -7.21 to -1.14), reported but not tested.

**Deviations.** None.

**What it shows.** Advice helps with a rule Fly B has never seen. When the rule returns to one Fly B already knew, Fly B recovers on its own and advice adds nothing. Strong teaching of the intermediate rule briefly slowed the return. Larger action spaces, noise and partial observability remain untested; with this cue, a second change to a *new* rule is impossible without changing the task.

## Study 1: does the connectome contribute?

**Method.** (E) Do Fly A's measured neurons carry its learned advice beyond the cue? A trained and an untrained Fly A saw identical inputs and noise; held-out decoders were compared against the cue alone, with a pre-set stop rule. (A-D) Teaching with spike advice, information-matched digital advice and random advice through the same spike channel, on the real connectome and on a rewired one (the simulator's built-in shuffled wiring). 5 diagnostic seeds; 15 teaching seeds.

**Seeds.** Diagnostic: 5 attempted, 5 completed. Teaching: 15 attempted; real wiring 15/15; rewired 0/15 (every rewired Fly A failed its gate).

**Results.** (E) Cue alone 0.939 average held-out accuracy; Fly A's pooled MBON rates 0.945; individual MBON counts at chance. Stop rule triggered. (A-D) Spike 65.60 and digital 65.60 on every seed (difference 0.00); trained minus random content +35.07 (CI +24.93 to +45.07); 100% of messages decoded. Rewired Fly A: 22-25 of 72 (chance is about 24) versus 60-70 on the real wiring.

**Deviations.** The first diagnostic run saved empty individual-MBON counts (a bug in the new script). It is kept unchanged. The fix and re-run were recorded in the protocol before the stop rule was applied.

**What it shows.** Fly A's advice is chosen by a software readout; its neurons carry what it senses, not what it learned. Spikes deliver the advice perfectly but add nothing beyond a digital message. The real wiring is needed for Fly A to learn from its senses at all, with the caveat that input scaling was fitted on the real wiring. Whether real wiring helps *teaching* could not be tested.

## Conclusions

**The strongest supported Synapse claim.** In this simulation, a few dozen bits of action advice from a teacher that has already adapted make a differently built student recover much faster after a rule change, and the benefit comes from the teacher's content. It beat matched random and shuffled advice in every relevant study here (+26.5 in Study 5, +35.1 in Study 1, +27.8 after change 1 in Study 4), on fresh seeds, under protocols fixed in advance.

**The most important negative result.** The connectome does not contribute anything to the teaching itself. Advice sent as spikes and as a plain number gave identical learning on 15 of 15 seeds, and Fly A's neurons carry no learned teaching signal beyond what it senses. The fly brain is the setting and the sensor, not the source of the advantage.

**What is actually new compared with established action advising.** The idea of a teacher sending action advice under a budget is not new (Torrey and Taylor 2013), nor is asking when uncertain (Ilhan et al. 2022), nor the risk of bad advice (Zhan et al. 2016). What this work adds is narrow and empirical: a controlled, pre-registered test of action advising between differently built learners inside a simulated fruit-fly connectome, with spike-level delivery, matched content controls and fresh-seed replication; and three measured findings for this system: timing within a fixed budget did not matter by round 48; confidence-based requests under-ask after a change; and where trust starts trades off protection from a bad teacher against speed with a good one.

**Did neural circuitry show a causal contribution?** Not to teaching or communication. Rewiring did stop Fly A from learning the task (chance-level scores on all 15 seeds), so the real wiring matters for Fly A's own learning, with the input-scaling caveat. But advice delivered without any neurons worked exactly as well.

**What we can safely tell HackPrinceton judges.**
- "A trained fly-brain model sending a few dozen bits of advice made a differently built student recover far faster after a rule change, and it was the advice's content, not just any message, that helped. We checked this on fresh seeds with rules fixed in advance."
- "The fly brain doesn't make the message special: a plain digital message works exactly as well. Rewiring the brain stopped the teacher from learning at all."
- "We also report what didn't work: 36 messages weren't reliably enough, letting the student ask cut messages but not enough, our trust rule protected against a bad teacher but slowed a good one, and advice didn't help when an old rule came back."
- Do **not** say: biological fly teaching, natural brain-to-brain communication, real-world fraud results from these studies, or that the connectome is needed for the advice to work.

**What remains unanswered.**
- The exact advice budget needed for reliable recovery (25-45 messages untested here).
- A request rule that uses correctness as well as confidence.
- A trust rule that keeps both protection and speed, for example by also trusting a teacher whose advice the student has not yet tested.
- Tasks with more actions, noisy outcomes, partial observability, or repeated changes to new rules.
- Whether rewiring stops learning because of the wiring itself or because of the mismatch in input scaling.
- Whether a student whose synapses actually learn would produce a natural teaching signal.

## Files created in this session

Protocols and reports in `docs/`: `advice-budget-replication-*`, `advice-timing-*`, `evidence-trust-*`, `repeated-change-*`, `connectome-contribution-2-*`, and this summary. Runners in `tools/`: `check_advice_budget_replication.mjs`, `check_advice_timing.mjs`, `check_evidence_trust.mjs`, `check_repeated_change.mjs`, `check_endogenous_signal.mjs`, `check_connectome_contribution_2.mjs`, with the matching `analyze_*.py` scripts. Raw results in `data/advice_budget_replication/`, `data/advice_timing/`, `data/evidence_trust/`, `data/repeated_change/`, `data/endogenous_signal/`, `data/endogenous_signal_v2/`, `data/connectome_contribution_2/`, with one `*_summary.json` per study. Every result file is write-once. No Git operations were run except the one approved pull at the start.
