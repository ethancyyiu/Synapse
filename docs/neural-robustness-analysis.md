# Frozen RPS decision margins under the fixed KC→MBON lesion

**Read-only saved-row analysis, .** The [locked feasibility study](neural-disruption-feasibility-protocol.md) attenuated the 9,210 KC→MBON edges into engineered MBON group 0 to 50% of their normal multiplier. Its [behavioral impairment gate failed](neural-disruption-feasibility-report.md). This analysis uses only the [saved intact, sham, and disrupted rows](../data/neural_disruption_feasibility.json); it does not change that result or run a simulator.

## Method and integrity

For each matched observed-cue round, the intact winning-action margin is the intact selected action's external Q score minus the highest of its other two Q scores. The disrupted margin uses that **same intact selected action**. A negative disrupted-minus-intact difference means the original decision became less secure. This is an action-score margin, not a biological neural decision threshold. The 69 observed-cue rounds per student are primary; the three cue-free opening rounds are excluded from context comparisons.

All 72 intact rows per student exactly match the archived final frozen Strategy B checkpoint in cue, action, opponent, reward, VPN route, pooled MBON rates, standardized features, and three Q scores. Sham rows equal intact rows exactly. On all 144 frozen rounds, selected actions equal the saved score argmax, and independently recomputed RPS rewards equal saved rewards. Both intact and disrupted conditions have **70/72 wins** per student; every observed cue has **23/23 wins**. No controller weights changed, according to the saved condition integrity flags. The earlier report also verified targeted synaptic multipliers and no untargeted multiplier changes.

## Paired action-score and margin changes

Scores below are external hybrid-controller Q values, in their arbitrary score units. Each action-score cell gives **mean signed change / mean absolute change** (disrupted minus intact) over 69 observed-cue rounds. Signed means can cancel substantial trial-level changes.

| Student | Rock Q | Paper Q | Scissors Q | Mean margin, intact → disrupted | Mean absolute margin change | Narrower / wider / same |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11015 | −0.051 / 0.171 | +0.103 / 0.200 | −0.017 / 0.178 | 2.003 → 2.003 | 0.314 | 32 / 33 / 4 |
| 11016 | −0.054 / 0.184 | +0.086 / 0.190 | −0.042 / 0.153 | 1.940 → 1.944 | 0.311 | 32 / 35 / 2 |

Thus **64/138** observed-cue margins narrowed without an action flip. The largest single-trial margin reductions were **2.034** and **1.412**; the smallest *disrupted* margins remained **1.005** and **0.934**, respectively. Every disrupted margin stayed positive. This directly explains the unchanged greedy decisions in these saved rows: the altered Q scores never reversed the top-ranked action. It does not prove immunity to other lesions, states, or policies.

## Context and action sensitivity

Under Strategy B, the correct observed-cue action equals the cue index; all observed-cue frozen actions were correct. Each cell below summarizes 23 matched trials. Rate change is the mean change in pooled group-0 MBON spikes per MBON. Margin change is disrupted minus intact. The three-cue action identity means context and selected action cannot be separated in this dataset.

| Student | Cue / selected action | Group-0 rate change | Mean margin change | Margin narrowed | Disrupted minimum margin |
| --- | --- | ---: | ---: | ---: | ---: |
| 11015 | 0 / rock | −0.221 | −0.216 | 20/23 | 1.005 |
| 11015 | 1 / paper | −0.130 | +0.242 | 1/23 | 1.913 |
| 11015 | 2 / scissors | −0.009 | −0.026 | 11/23 | 1.894 |
| 11016 | 0 / rock | −0.180 | −0.162 | 18/23 | 0.934 |
| 11016 | 1 / paper | −0.112 | +0.217 | 0/23 | 2.142 |
| 11016 | 2 / scissors | −0.025 | −0.043 | 14/23 | 1.199 |

The target group's **overall** mean rate fell 0.480→0.360 (25.0%) and 0.470→0.364 (22.5%), as previously reported. Its effect on Q scores depends on the learned cue-specific readout coefficients and simultaneous changes in the other two MBON groups. The saved final linear head weights and standardized features permit an exact fixed-runner-up decomposition of each margin change; the runner-up action did not switch in any observed-cue row. On cue 0, the mean group-0 feature contribution to the margin change was **−0.205/−0.153** for students 11015/11016; on cue 1 it was **+0.221/+0.210**; on cue 2 it was **−0.012/−0.030**. Contributions from the other MBON features account for the remaining changes. These are arithmetic contributions within the saved external linear readout, not a claim that all effects arise solely from group 0.

The largest mean *absolute* margin variation occurred on cue 2 (**0.437/0.477**), despite its small mean signed change and small mean target-rate change. Cue 0 showed the clearest systematic narrowing; cue 1 became more secure. All three contexts retained the same decisions. There is no separate action sensitivity estimate independent of context, because each cue led to only its correct action in these frozen evaluations.

## Interpretation and limits

For this **particular 50% synaptic attenuation**, in these **two already trained Strategy B student networks** and their **saved 72-round frozen streams**, neural activity and external Q scores changed, but the selected action and win total did not. This supports a narrow quantitative observation of decision-margin robustness: all 138 observed-cue disrupted margins stayed positive even though 64 narrowed. It does **not** establish general circuit robustness, a dose-response curve, tolerance to stronger lesions, recovery through learning, or relevance to real neurological injury. The locked behavioral gate still fails and there is no impaired policy to recover under this manipulation.

Different lesion strengths, targets, independently trained networks, noisy or unfamiliar contexts, and prospective frozen tests would be needed to generalize. None were examined here. Given the failed behavioral gate and the HackPrinceton timeline, further robustness experiments are **not warranted before the demonstration**; preserve this informative negative result.

**Read-only reproduction:** parse the saved JSON, pair rows by student/episode/round, compute `scores[action] − max(scores[other actions])` in intact and disrupted conditions, and compare their difference for the 69 non-null cues per student. The original archived final checkpoints are in the [initial](../data/rps_action_preference_strategy_b_initial.json) and [confirmation](../data/rps_action_preference_strategy_b_confirmation.json) Strategy B artifacts. Do not run the write-once feasibility runner for this analysis.
