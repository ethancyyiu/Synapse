# Superseded: teacher-richness next test

This note predates the action-preference experiments and must not be used as the current research protocol. Its one-bit bottleneck hypothesis and proposed richer-message experiment describe the older proposal/veto channel, not the newer three-action preference channel.

## Current evidence

- The engineered post-sensory action-preference system passed locked early, frozen, teacher-disconnected comparisons on two fixed deterministic RPS rules. Four Strategy A students and two Strategy B students passed their respective control gates. See the [Strategy A audit](action-preference-independent-audit.md), [Strategy B audit](action-preference-strategy-b-independent-audit.md), and [HackPrinceton readiness handoff](hackprinceton-final-readiness.md).
- The student plays its current action before receiving the preference; it then applies an engineered decoded-action imitation update alongside its own-reward update. KC→MBON plasticity remains frozen.
- In the separate Strategy B ablation, measured DN-spike delivery and direct digital delivery produced identical actions, updates, weights, and frozen predictions. The spikes carried the preference reliably, but no advantage over digital delivery was observed. See the [ablation report](connectome-contribution-ablation-report.md).
- The older one-bit proposal/veto experiment remains a separate failed result. It is not overturned by the action-preference result.

## Research decision

Do not run a richer-message experiment on the basis of this superseded hypothesis. A noisy-opponent extension is also not currently authorized by the existing evidence: earlier noisy receiver and decoder prerequisites failed, and the lab notebook explicitly stops that route absent a new rationale. The synthetic sensor-classification line also stopped at its first neural-information gate and is deferred until after HackPrinceton.

For the immediate project, use the archived, audited demo route and complete the presentation-machine rehearsal in the [demo guide](demo-guide.md). After the event, any generalization study must begin with a new prospective task protocol and a no-message feasibility/headroom gate. Do not start a teacher-student comparison unless that prerequisite passes; keep the trained, untrained, shuffled, generic, no-advice, and where relevant direct-digital controls fixed in advance.

## Historical result

The original archived Strategy A proposal/veto test in `data/rps_teacher_specific_speed_initial.json` had `first58` values `{ correct: 12, alone: 96, untrained: 12, shuffled: 12, generic: 12 }`. At round 12, the frozen scores were `{ correct: 69, alone: 24, untrained: 68, shuffled: 69, generic: 69 }`; at round 288, all five were 69/72. This historical result remains valid for that older mechanism and is not the current action-preference result.
