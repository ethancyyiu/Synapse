# Locked 20-pair Strategy B action-preference cohort

Locked , before any cohort simulation. This re-runs the unchanged architecture and gates of the [Strategy B protocol](action-preference-strategy-b-protocol.md) on 20 new teacher/student pairs, to replace six example students with a measured effect size and uncertainty. It tests one thing: sample size. No architecture, parameter, threshold, or budget changes.

## Identities

Pair `i` (0 to 19): teacher seed `21001 + 2i`, student seed `21002 + 2i`. Training opponent seeds `83000 + 12i` through `83011 + 12i`. Evaluation opponent seeds: scanning upward from `93000 + 100i`, the first seed for each of initial moves 0, 1, 2. This selection uses opponent code only. Shuffled and generic control streams use `2026110902 + 2i` and `2026110903 + 2i`.

No pair is replaced or rerun. Each pair writes one write-once file under `data/rps_action_preference_cohort/`.

## Per-pair procedure

Identical to the Strategy B protocol: teacher, receiver, synthetic, independent-learning, and source gates, then five arms (`trained`, `alone`, `untrained`, `shuffled`, `generic`) with frozen checkpoints at 0, 6, 12, 24, 48, 96, 288. A pair that stops at a prerequisite gate is reported as stopped and is not an effect-size observation.

## Cohort analysis, fixed in advance

For each completed pair, `advantage = early(trained) - max(early(alone), early(untrained), early(shuffled), early(generic))`, where `early` is the mean frozen wins/72 over checkpoints 6, 12, 24, 48.

Report: number of pairs completed and stopped by reason; mean advantage with a 95% percentile bootstrap interval (10,000 resamples of pairs, `rng(2026110901)`); pairs meeting the original per-pair pass; final 288-round wins by arm.

**Cohort pass:** at least 15 pairs complete, the bootstrap lower bound is above +4/72, and at least 75% of completed pairs meet the original per-pair pass. Anything else is reported as a failed cohort.

## Limits

A pass supports early retained learning speed for this engineered hybrid on fixed Strategy B only. It does not show final-accuracy benefit, a causal role for the connectome, generalization to other opponents, or biological teaching.
