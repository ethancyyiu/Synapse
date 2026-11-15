# Synapse advice budget and timing study

Locked before any run of this study. This is a **new**, separate test; the saved rule-switch, fault-response, and banking results remain unchanged. The existing rule-switch cohort used one advice opportunity on each observed-cue round. It cannot identify the minimum useful advice budget because later checkpoints also include more student actions and rewards.

## Question and hypothesis

On the existing RPS rule-switch task, does a trained teacher sending only 12 action preferences over 48 training rounds help Fly B at the frozen 48-round checkpoint, beyond no advice and 12 equally timed random preferences? Does timing the same 12 messages early versus spread across the 48 rounds change the outcome? This tests a fixed budget, **not** a universal minimum.

## Fixed design

- Preserve the existing connectome, teacher and student controllers, learning equations, teacher gate (at least 58/72), opponent rules, 24-round episodes, 72-game teacher-disconnected evaluation, and paired random streams. No API calls.
- One pilot seed `99` checks code and runtime only. Its outcome cannot change this protocol or enter the final estimate. Final seeds are `100` through `114`, with no replacement. The existing seed formulas apply to these new indices.
- Two post-switch episodes give 48 training rounds and exactly 46 rounds with an observed cue. A message names one of three actions using the existing two-neuron spike code. The teacher preference is computed on every round, but is transmitted only at assigned opportunities.
- Arms: `alone` (0 messages), `full` (all 46 opportunities), `early12` (eligible opportunity indices 0–11), `spaced12` (indices `floor((i + 0.5) * 46 / 12)` for `i=0..11`), and `random12` (random actions at the same opportunities as `spaced12`). The three 12-message arms have the same nominal message count. All learners receive their own action rewards on every round.
- Evaluation checkpoints: before training, after 24 rounds, and after 48 rounds. The primary endpoint is frozen wins out of 72 at round 48. Save actual message and decode counts, teacher gate result, and all arm scores for every seed. Report the fraction of gated seeds; never silently discard a failed teacher.

## Analysis fixed in advance

Primary contrast per completed seed: `spaced12 - max(alone, random12)` at round 48. Report the mean and a paired 95% percentile bootstrap interval (10,000 resamples, fixed seed `2026111512`). A *positive* result requires at least 10 of 15 seeds to pass the teacher gate, all 12 scheduled messages to decode in each completed seed, mean `spaced12` at least 58/72, and the bootstrap lower bound on the contrast to exceed 4 wins. Otherwise the gate fails. This threshold is a study decision, not an expected effect.

Secondary, descriptive contrasts: `early12 - spaced12`, `full - spaced12`, and the 24-round scores. Do not select the best arm, threshold, or checkpoint after seeing results. No claim about the smallest sufficient budget follows even if 12 messages pass, because budgets 1–11 are untested. A failed 12-message test also does not prove that more than 12 are necessary.

## Interpretation limits

Prior action-advising work already studies message budgets and timing. This test contributes a controlled estimate for this particular engineered fly-connectome simulation. The spike code is software-driven, not naturally emitted by Fly A, and the test cannot establish biological teaching, real-world benefit, or a neural advantage over an information-matched digital message.
