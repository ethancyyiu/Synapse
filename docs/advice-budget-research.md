# How much advice does Fly B need?

This note separates published evidence, the existing Synapse results, and the new fixed-budget experiment in [the protocol](advice-budget-protocol.md). Scores below are from simulated RPS rules, not biological learning or a banking deployment.

## How one short message can matter

Fly B chooses an action before receiving a message. It then observes its own game reward and updates its own learner. If Fly A's preferred action was decoded from the engineered two-neuron spike code, Fly B also makes a small imitation update toward that action in the current cue-specific head (`PreferenceStudent.imitate` in `js/rps_challenge_students.js`). The message supplies one of three action names, not Fly A's model weights or past examples. Repeated updates can accumulate into a different policy; advice from an outdated teacher can instead push the policy the wrong way.

The current decoder distinguishes three codewords, so the *decoded content* of one message is at most `log2(3) ≈ 1.585` bits. This does not mean the overall interaction is one or two bits: the existing 48-round comparison offered advice on 46 rounds with observed cues. The spike generator is software-driven by Fly A's selected preference; the experiment does not show spontaneous teacher-to-student neural communication.

## What is already measured

In the locked 15-seed rule-switch cohort, every completed run decoded 92 trained messages over 96 rounds. Each 24-round episode begins without a usable cue, so the 24-, 48-, and 96-round checkpoints correspond to 23, 46, and 92 advice opportunities. Recomputed directly from `data/rule_switch_cohort/seed_*.json`:

| Student training rounds | Advice opportunities | Mean taught wins / 72 | Mean alone wins / 72 | Runs with taught score ≥58/72 |
|---:|---:|---:|---:|---:|
| 24 | 23 | 51.33 | 4.33 | 4/15 |
| 48 | 46 | 67.73 | 10.07 | 15/15 |
| 96 | 92 | 69.93 | 35.13 | 15/15 |

These checkpoints **do not reveal the minimum advice count**. Between checkpoints, B receives more advice *and* takes more actions and observes more rewards. The existing cohorts did not randomize advice frequency at a fixed number of training rounds. A 72-game frozen test at each checkpoint establishes what B learned by then, without help during that test.

## Relevant prior work

- [Torrey and Taylor, *Teaching on a Budget* (AAMAS 2013)](https://myslu.stlawu.edu/~ltorrey/papers/aamas13.pdf) studied action advice under a limited budget. The same number of suggestions had different effects when sent at different times, including for learners with different methods or representations. Thus budgeted advice itself is established prior art.
- [Ilhan et al., *Methodical Advice Collection and Reuse* (ALA 2022)](https://ala2022.github.io/papers/ALA2022_paper_43.pdf) studied student uncertainty as a trigger for requesting advice. Asking only when uncertain is a plausible future comparison here, not a novel mechanism by itself.
- [Zhan et al., *Policy Advice from Multiple Teachers and Negative Transfer* (IJCAI 2016)](https://www.ijcai.org/Proceedings/16/Papers/330.pdf) formalized how poor or mismatched advice can harm learning. Synapse's stale-teacher failure is consistent with that risk; it does not establish a new general law.

No paper above gives a universal number of messages for this specific Synapse learner. The number depends on the task, how many cues and actions there are, teacher quality, reward frequency, student update strength, and which rounds receive advice.

## New controlled test

The separate [locked protocol](advice-budget-protocol.md) holds the 48 student training rounds and frozen evaluation fixed while varying message count and schedule. It compares 0 messages, 12 early, 12 spaced, 12 random at the same spaced positions, and all 46 opportunities on new seeds. Pilot seed 99 is for code/runtime checking only and is excluded from the final estimate. The new study can test whether **12 spaced messages suffice under its preset criterion**. It cannot identify a global minimum because budgets 1–11 and other schedules remain untested.

## Final result (15 new seeds; pilot excluded)

All 15 trained teachers passed the preset gate, all scheduled messages decoded through the DN spike channel, and every student was evaluated without the teacher. At the fixed 48-round endpoint:

| Advice condition | Messages in 48 rounds | Mean frozen wins / 72 |
|---|---:|---:|
| Learn alone | 0 | 12.20 |
| Random advice, spaced | 12 | 13.67 |
| Trained advice, early | 12 | 41.20 |
| Trained advice, spaced | 12 | 43.73 |
| Trained advice, every opportunity | 46 | 68.07 |

The prespecified paired contrast, spaced trained advice minus the stronger of the alone and random-advice controls *within each seed*, averaged **+28.87 wins / 72** (95% paired percentile bootstrap interval **+20.13 to +37.33**). Thus twelve meaningful messages had a measurable effect in this simulation. **The prespecified success criterion FAILED**: the spaced-12 mean was 43.73/72, below the required 58/72. Only 2/15 spaced-12 runs reached 58/72. Full advice reached a mean of 68.07/72. This is evidence that 12 spaced messages helped but did not reliably yield recovery by round 48 under the locked criterion; it does not establish a minimum sufficient budget.

Timing is unsettled. At round 24, early-12 averaged 21.93/72 versus 10.80/72 for spaced-12, but by round 48 the spaced mean was only 2.53 wins higher than early. These are descriptive comparisons, not a selected winning schedule. The study used a fixed tape of preferences generated by each trained teacher, then replayed that same content to matched arms through the simulated DN spikes. That isolates message timing and count; it does not test a teacher adapting its advice to each student's state. Full per-seed outputs are in `data/advice_budget/seed_100.json` through `seed_114.json`, and the aggregate is in `data/advice_budget_summary.json`.
