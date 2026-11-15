# Study 2: when should 36 messages be sent? (timing and student requests)

Locked before the pilot and before any final run. New and separate; the advice-budget studies, the adaptive sparse-teaching pilot, and every Science/banking artifact stay unchanged.

## Prior evidence

- At 48 rounds the advice-budget follow-up found 36 spaced trained messages to be the smallest tested budget meeting its recovery criterion (64.47/72, 13/15), on seeds already used before. Study 5 checks it on fresh seeds.
- Timing was barely measured: 12 early versus 12 spaced messages differed by 2.53 wins at round 48 (descriptive only). The follow-up's planned timing/request comparison (Part 2) was never run.
- The earlier 288-round request pilot (`adaptive-sparse-teaching-report.md`) used an experience-count rule. Its trained-request arm beat a fixed schedule but not its exposure-matched **shuffled** control, so it failed its teacher-specific gate. Any timing advantage here must therefore be checked against content controls delivered at the same moments.
- Budgeted advice and uncertainty-triggered requests are established ideas (Torrey and Taylor 2013; Ilhan et al. 2022). This study measures them in this simulation; it does not claim a new mechanism.

## Hypotheses

- **H-timing.** At a fixed 36 messages, timing changes the round-48 frozen score: early, evenly spaced and randomly timed schedules differ.
- **H-request.** Fly B asking for advice when it is unsure reaches the same recovery with **fewer** than 36 messages, and the benefit comes from the teacher's content, not just from extra messages at those moments.

## Design

Same task, teacher, student, learning rule, message code and evaluation as the follow-up: rule-A history, 48 rule-B rounds (46 observed-cue opportunities), own-action reward every round, measured DN spike decoding, teacher-disconnected 72-game frozen checks at 0, 24 and 48 rounds. Runner: `tools/check_advice_timing.mjs`. Teacher `26001 + 2s`, student `26002 + 2s`, all follow-up seed formulas.

Arms (all from the same trained-teacher preference tape):
- `alone` (0 messages) and `full46` (all 46 opportunities, reference).
- **A** `early36`: opportunities 0-35.
- **B** `spaced36`: opportunities `floor((i + 0.5) * 46 / 36)`, the follow-up's schedule.
- **C** `randomTime36`: 36 of the 46 opportunities chosen by Fisher-Yates with seed `2026111600 + s`, fixed before training.
- **D** `requested`: Fly B requests advice (rule below), at most 36 messages, unused budget never backfilled.
- **E** `requestedRandom`: a random action at exactly the moments D requested (yoked).
- `requestedShuffled` (extra content control): the trained preferences permuted across the 46 opportunities (seed `2026111700 + s`), delivered at exactly the moments D requested.

**Request rule.** On an observed-cue round, Fly B requests advice if the last time it saw this same cue, the gap between its two highest decision scores was below a threshold `tau` (or it has not seen this cue since the change), and fewer than 36 messages have been sent. This uses only Fly B's own earlier state. It never uses the correct action, future rewards, or Fly A's preference. The request is a one-bit digital signal outside the simulated neural channel; the advice itself is still sent as measured DN spikes. Why "last time": in this simulator the decision and the advice happen in the same step, so a same-round request would need a change to shared code; the previous same-cue gap is the most recent uncertainty Fly B can know without one.

**Choosing `tau` (pilot only, seed 299, excluded from all results).** Start at `tau = 0.20`, the value the follow-up pre-registered for this rule. Keep it if the pilot's request count is between 12 and 34. If the pilot hits the 36 cap, halve `tau` once (0.10); if it requests fewer than 12, double it once (0.40); then keep that value whatever it gives. The locked value is written into the runner before any final seed. Pilot results never influence anything else.

## Seeds

Pilot: `s = 299`. Final: `s = 300-314` (teacher seeds 26601-26629), never used by any study. No seed is replaced; a teacher that fails its gate is a teacher-readiness failure.

## Fixed analysis

`tools/analyze_advice_timing.py`, written before any final run. Paired differences in frozen wins out of 72 at round 48, each with a 95% percentile bootstrap (10,000 resamples, fixed seed `2026111560`).

**H-timing.** `early36 - spaced36` and `randomTime36 - spaced36`. A timing difference is shown only if an interval excludes zero; otherwise timing is reported as no detectable difference at this sample size. No "optimal schedule" is claimed unless one schedule beats both others with intervals excluding zero.

**H-request.** Supported only if all hold: (1) mean messages used by `requested` is below 36; (2) `requested` recovers (mean at least 58/72 and at least 12/15 students at or above 58/72); (3) it is not worse than `spaced36`: lower bound of `requested - spaced36` above -4; (4) the content matters: lower bound of `requested - max(requestedRandom, requestedShuffled)` above 4. If (1)-(3) hold but (4) fails, report that requests saved messages but the benefit was not shown to come from the teacher's content.

**Reported, not tested:** every arm's mean and recovery count at 24 and 48 rounds; actual messages and decodes per arm; `full46` versus each 36-message schedule; when requests happened (early versus late).

At least 10 of 15 teachers must pass their gate for any claim. Nothing changes after final results are seen.

## Limits

One task, one change, one budget. The request signal is digital and uses the previous same-cue uncertainty, not a calibrated probability or a neural return path. Results apply to this simulation only.

## Pilot outcome (recorded before any final seed)

Pilot seed 299 with `tau = 0.20`: teacher 68/72, 12 requests (within 12-34), all messages decoded. By the rule above, `tau = 0.20` is locked for seeds 300-314. Pilot scores are not used anywhere else.
