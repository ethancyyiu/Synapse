# Exploratory noisy RPS demo feasibility — locked before the pilot

This is a new demo feasibility check, separate from all archived teaching and noisy-receiver gates. No historical result changes. The proposed opponent follows Strategy A from the *observed* previous opponent move, with `noise=0.3` in the existing opponent. Thus the base next move has probability 0.8 and each other move 0.1. The hidden random draw and next move are never controller inputs. The best memoryless action wins with probability 0.8, versus 1.0 for deterministic Strategy A.

One fixed pilot: teacher network/controller seed **931**; matched student seed **10931**; opponent streams **41000–41011** for teacher training, **51000–51002** for its frozen test, **41000** for student training, and **51000–51002** for the matched student frozen test. Keep the existing three-MBON teacher controller, cue-specific student controllers, DNge175 spike interface, initial zero student heads, 12 × 24 teacher rounds, 24 student preview rounds, and 72-round frozen disconnected evaluation. No architecture, reward, timing, exploration, stimulation, or learning-rate change. Run once and save one write-once artifact.

**Prerequisites, applied in order:** (1) seeded opponent replays and both student streams match, and all three previous-move cues occur; (2) teacher frozen test scores at least **45/72 wins**, at least **half the trials in each observed cue**, and selects each action at least **six times** across observed-cue trials; (3) all actual sent DN messages decode to the teacher preference, no student cue-route mismatch occurs, and KC→MBON weights stay frozen. Stop at the first failed prerequisite. The earlier external MBON-centroid receiver failure is unchanged; this pilot uses the existing VPN-derived route and does not validate that older decoder.

The exploratory display gate is **With Teacher ≥ Alone +8/72** after 24 rounds, both evaluated frozen without communication. This is a one-seed, two-arm demonstration gate, **not** a teacher-specific research claim: shuffled, untrained, and generic controls are absent. A pass would permit adding a clearly labeled opt-in *Noisy Strategy A — experimental* opponent choice to the live Challenge; a failure means keep it out of the main selector. No parameter or seed replacement after viewing the outcome.

## One fixed pilot result

The first execution computed the pilot but the stale workspace sandbox denied Node permission to create the result file in the renamed repository. It produced no saved result or console metric. The **same seed and settings** were rerun once with write permission; no parameter was changed. The resulting [write-once artifact](../data/rps_noisy_demo_feasibility.json) contains the 24 measured student messages and summary scores. Reproduce only in a copy without that file using `node tools/check_rps_noisy_demo_feasibility.mjs`.

| Gate or measure | Result |
| --- | --- |
| Matched seeded opponent streams | Pass; independent of student action |
| Fly A frozen wins | **50/72**; cue 0 **17/23**, cue 1 **13/25**, cue 2 **19/21** |
| Fly A observed-cue action counts | Rock **15**, Paper **23**, Scissors **31** |
| Teacher-learning gate | Pass (≥45 wins, ≥half per cue, ≥6 selections per action) |
| Measured DN messages | **23/23** cue-bearing rounds decoded; first cue-free round had no message |
| Frozen disconnected wins after 24 student rounds | Alone **39/72**; With Teacher **58/72**; difference **+19/72** |
| Exploratory display gate | Pass (predeclared ≥+8/72) |

This is a feasible **experimental demo option**, not an independent teaching result. One seed cannot establish reliability or a trained-teacher-specific effect; matched shuffled, untrained, and generic controls were not run. In particular, the stochastic opponent makes the theoretical best win probability 80%, so deterministic-task thresholds and archived claims must not be reused. The earlier noise-0.5 pooled-MBON recognition failure remains failed and is not repaired by this task. The original deterministic presets remain the primary validated presentation.
