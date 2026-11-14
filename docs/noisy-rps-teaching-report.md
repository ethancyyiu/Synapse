# Noisy Strategy A teacher-specific early-learning test

**Status: failed the prospectively locked all-seed gate.** [Protocol](noisy-rps-teaching-protocol.md) was written before the focused correctness check and single experiment. The [write-once result](../data/rps_noisy_teaching.json) stores all training and frozen evaluation rows. Reproduce only in a copy without that artifact: `node tools/check_rps_noisy_teaching.mjs check`, then `node tools/check_rps_noisy_teaching.mjs run`. No demo, controller, stimulation, neural dynamics, or historical artifact was changed for this study.

The fixed teacher seed 931 exactly reproduced its prior **50/72** frozen wins. Its 24 preference messages exactly matched the earlier noisy demo pilot. The causal shuffle changed **12/23** cue-bearing preferences. Matched opponent schedules, identical initial student weights and frozen **22/72** scores, frozen weight integrity, VPN cue routing, and decision-before-outcome order all passed. Every delivered preference decoded from measured DN spikes: **23/23 per advice arm per seed**, **207/207** total. Silence and frozen evaluations had no message, imitation, exploration, or learning. KC→MBON weights stayed frozen. The shuffled control drew only from prior teacher preferences, so no future opponent move or future teacher message could reach an earlier decision.

| Student | Trained at 12 | Strongest control at 12 | Gap | Trained at 24 | Strongest control at 24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12341 | 57/72 | 44/72 (shuffled) | +13 | 58/72 | 58/72 (alone) |
| 12342 | 58/72 | 58/72 (alone) | 0 | 58/72 | 58/72 (alone, shuffled) |
| 12343 | 58/72 | 58/72 (shuffled) | 0 | 58/72 | 58/72 (alone, shuffled) |

Full frozen curves, in checkpoint order **0, 12, 24**:

| Student | Trained | Alone | Shuffled | Untrained |
| --- | --- | --- | --- | --- |
| 12341 | 22, 57, 58 | 22, 39, 58 | 22, 44, 57 | 22, 21, 27 |
| 12342 | 22, 58, 58 | 22, 58, 58 | 22, 44, 58 | 22, 22, 22 |
| 12343 | 22, 58, 58 | 22, 37, 58 | 22, 58, 58 | 22, 22, 22 |

The primary criterion required trained advice to beat the strongest control by **at least 4/72 at round 12 for every student**. Only 12341 passed individually; the all-seed experiment **failed**. The early advantage in the prior single-seed demo pilot did not generalize to this three-seed controlled comparison. The noisy option remains an **experimental live challenge**, not validated teacher-specific teaching evidence. The deterministic Strategy A/B results and all earlier noisy-receiver failures retain their original interpretation. Stop this study without changing the opponent, controls, or learning rule.
