# Study 4: does teaching survive a second rule change?

Locked before any run of this study. New and separate; every earlier study and every Science/banking artifact stays unchanged.

## Why this task

Every confirmed teaching result so far has one controlled change: the Rock-Paper-Scissors rule switch and the fault-response task. Earlier attempts to go further either stopped before teaching (sensor readings, warehouse route, package recovery) or failed their rule (the noisy-opponent test; spike loss was a channel test, not a task test). None tested **repeated** change.

Options considered before any result, and why this one:
- **Larger action space, partial observability, delayed or noisy reward**: each needs a new task, a new cue route, or a change to the learner. Not the smallest modification.
- **A stochastic opponent**: already tested once (noisy Strategy A) and failed.
- **Repeated changes**: needs no code change to the learner or the shared simulator, and tests an important limit, whether advice still helps when Fly B has to unlearn twice and the second change collides with what it was just taught.

With the existing cue (the opponent's previous move), the only non-trivial deterministic rules are offsets 1 (rule A) and 2 (rule B); offset 0 makes the opponent repeat one move forever. So the repeated change is **A, then B, then back to A**.

## Hypothesis

After the second change (B back to A), a Fly B advised by the teacher that is current for each phase leads the best control by more than 4 wins out of 72, 48 rounds after the second change.

## Design

Runner: `tools/check_repeated_change.mjs`, built from the rule-switch machinery with no change to the learner, the message code, or the evaluation.
- Fly B: four 24-round episodes of rule A history (as in the rule-switch studies), then 48 rounds of rule B (change 1), then 48 rounds of rule A (change 2). Own-action reward every round.
- Teachers: two teachers on the same network seed, trained by the existing procedure, one on rule B and one on rule A; each must reach 58/72 on its own rule. The rule-B teacher advises during the rule-B phase and the rule-A teacher during the rule-A phase ("current teacher"). This tests the student side of repeated change, not a teacher that must re-learn by itself.
- Advice: the current teacher's preferred move on every observed-cue round, sent as measured DN spikes, as in the rule-switch cohort.
- Arms: `alone` (no advice); `trained` (current teacher); `random` (a random move at the same times); `shuffled` (the current teacher's moves permuted among observed-cue rounds within each phase). Direct digital advice is left out: earlier ablations found it identical to spikes, so it adds nothing to the question of task generalization.
- Frozen, teacher-disconnected 72-game checks at 0 and 48 rounds (on rule B), and at 72 and 96 rounds (on rule A, i.e. 24 and 48 rounds after the second change).

## Seeds

New formulas: teacher `33001 + 2s`, student `33002 + 2s`; opponent bases 1110000 (rule-B teacher training), 1130000 (rule-A teacher training), 1150000 (history), 1170000 (rule-B gate and checks), 1180000 (rule-A gate), 1190000 (switch phases), 1200000 (rule-A checks); shuffle seeds `2026111800 + s` and `2026111900 + s`. None are used by any other study. **Implementation check:** `s = 20` (excluded). **Final:** `s = 0-14`. No seed is replaced. A seed whose rule-B or rule-A teacher misses 58/72 is a teacher-readiness failure.

## Fixed analysis

`tools/analyze_repeated_change.py`, written before any final run. Per completed seed, `advantage = trained - max(alone, random, shuffled)` in frozen wins out of 72 at round 96 (48 rounds after the second change). Mean with a 95% percentile bootstrap (10,000 resamples, fixed seed `2026111540`).

**Pass:** at least 10 seeds complete and the bootstrap lower bound exceeds 4. Otherwise **fail**. If fewer than 10 seeds complete because teachers miss their gates, the study is reported as **stopped at teacher readiness**, not as a teaching result.

**Reported, not tested:** the same advantage at round 48 (after change 1, as a check against the rule-switch result) and at round 72; mean wins per arm at every checkpoint; seeds with advantage at most 4; `alone` at 96 (how fast Fly B recovers an old rule alone).

Nothing changes after final results are seen.

## Limits

Two rules and two changes on one simulated task, with fixed cues and outcomes. A pass would show the effect survives returning to an earlier rule here, not under noise, larger action spaces, or partial observability. If Fly B recovers the old rule quickly on its own, advice may add little; that would be a real finding about when advice is unnecessary, not a defect to fix.
