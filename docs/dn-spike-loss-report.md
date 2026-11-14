# DN spike-loss teaching pilot

**.** This [prospectively fixed protocol](dn-spike-loss-protocol.md) thinned actual measured DN3141/3165 spike events by 0%, 10%, or 30% **after the unchanged simulator step and before the fixed count decoder**. The first execution completed all comparisons but Windows denied the final write-once JSON creation; an identical technical replay was run solely to save the result, with no protocol, seed, loss, code, or gate change. Historical experiments and the RPS demo were untouched.

The source was the archived reward-trained Strategy B teacher preference tape (teachers 11013/11014). New student seeds were **23001/23002**. Each student used matched starting weights, 12×24 opponent rounds and frozen 72-round held-out evaluations at 0/6/12/24/48/96/288. The digital arm received precisely the same decoded preference or erasure as its paired neural arm at the same time, but had no DN drive. Shuffled advice used the previously fixed permutation; silence had no imitation. All arms retained their own reward learning. A dedicated spike-loss RNG did not alter neural, opponent, or exploration streams.

## Results

Early is the mean of frozen wins/72 at rounds 6, 12, 24, and 48. Final is frozen wins/72 at round 288, with the teacher disconnected.

| Student | Spike loss | Trained neural early / final | Matched digital early / final | Shuffled early / final | Silence early / final | Trained DN codes correct | Measured spikes retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23001 | 0% | 69.5 / 70 | 69.5 / 70 | 50.5 / 40 | 29.5 / 70 | 276/276 | 2868/2868 |
| 23001 | 10% | 69.5 / 70 | 69.5 / 70 | 50.5 / 40 | 29.5 / 70 | 276/276 | 2589/2868 |
| 23001 | 30% | 69.5 / 70 | 69.5 / 70 | 50.5 / 40 | 29.5 / 70 | 276/276 | 2013/2868 |
| 23002 | 0% | 71 / 70 | 71 / 70 | 71 / 69 | 53.5 / 70 | 276/276 | 2849/2849 |
| 23002 | 10% | 71 / 70 | 71 / 70 | 71 / 69 | 53.5 / 70 | 276/276 | 2572/2849 |
| 23002 | 30% | 71 / 70 | 71 / 70 | 71 / 69 | 53.5 / 70 | 276/276 | 1970/2849 |

The trained-neural curves at **0/6/12/24/48/96/288** were **24/70/70/68/70/70/70** for 23001 and **24/71/71/71/71/70/70** for 23002 at every tested loss level. The matched digital curves were identical. At final disconnection, trained-neural students selected all three actions on **23/23/23** and **22/23/24** observed-cue rounds and won every observed-cue round. The remaining cue-free openings account for the final 70/72.

The measured-spike channel decoded all **1,656/1,656 trained messages** across both students and three loss rates; there were no erasures or wrong action codes. The shuffled neural channel likewise delivered every intended shuffled code. At 30% thinning, the trained channel retained about 70% of its actual DN spikes, yet enough spikes remained in every required target for the binary presence code. This shows robustness of this **engineered count code under independent per-spike loss**, not tolerance to whole-message erasure, correlated neuron loss, different noise models, or reduced stimulation.

The prospectively required **teacher-specific** early advantage over both shuffled advice and silence passed for 23001 but **failed for 23002 at all three loss rates**: trained and shuffled both had early mean **71/72**. Thus the all-student survival criterion fails already at 0% loss, and no teacher-specific advantage under corruption is established. Digital and neural arms had identical decoded inputs, training actions, rewards, final controller weights, and frozen curves; neural delivery did **not** beat matched digital communication. The 30% result cannot be called an improvement in learning or a communication-efficiency gain.

## Integrity, limits, and reproduction

The focused check verified that the same measured DN spikes decode normally at zero thinning and erase at a diagnostic 100% thinning, without changing simulated spikes or synaptic multipliers. The fixed-weight frozen score calculation reproduced all 72 archived final scores and actions **from their saved neural features** for the earlier reference student. The new frozen neural feature tapes were generated with the same reset, cue drive, and keyed-stream rules; their equivalence to separate per-arm simulations follows from the fixed neural weights and action-independent opponent, rather than a second simulation check. Within each new student and loss level, all arms had identical initial controller weights, cue/opponent streams, and neural synaptic multipliers; frozen evaluation used no DN input, exploration, reward update, or imitation. Realized rewards were allowed to differ when actions differed. No failed historical gate was reinterpreted.

The [write-once artifact](../data/dn_spike_loss.json) preserves the full training and frozen action rows, counts, controller weights, curves, and checks. Its embedded protocol SHA-256, `990bfa55471f229ab832bd10f0dfc8ec8b5483a1db847bdfde41c13a79e02c10`, matches the locked file. The reference [connectome ablation artifact](../data/rps_connectome_contribution_ablation.json) supplies the previously fixed teacher and shuffled tapes. Reproduce only in a copy without the write-once result: `node --check tools/check_dn_spike_loss.mjs`; `node tools/check_dn_spike_loss.mjs`. No additional loss schedule, seed, or neural interface was tested.
