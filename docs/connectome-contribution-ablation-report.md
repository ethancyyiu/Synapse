# What the simulated connectome contributes to RPS teaching

> Archive note: Calendar dates were removed from this working tree after the experiment. The hashes below refer to the original research files; they do not authenticate the edited files.

**Result:** In one locked, two-student Strategy B ablation, the measured DN-spike and matched direct-digital teacher-preference arms produced **identical training actions, rewards, controller weights, and all frozen evaluation actions**. Both learned earlier than silence and a globally scrambled neural-message control on the predefined early measure. The spike channel transmitted usable learned information with no observed fidelity or learning cost; this study does **not** show that DN downstream connections or the detailed connectome are needed for the teaching advantage.

The [prospective protocol](connectome-contribution-ablation-protocol.md) contains the pathway diagram, fixed seeds, controls, gates, and stopping rules. Its SHA-256 (`4DD5BE212C272AD93CCFF2DD4DBFCB988BD47AF510D524A227AD67883B95D9EA`) matches the hash embedded in the [write-once result](../data/rps_connectome_contribution_ablation.json). Local mutable files alone cannot independently certify the time of locking.

## Actual causal pathway

`Public prior RPS move → simulated Fly A VPN/KC/MBON spikes → external reward-trained Q readout → preferred action → directly driven DN3141/3165 spikes in Fly B → separate measured-count buffer → software three-code decoder → Fly B external cue-head imitation update → frozen disconnected policy.` Fly B separately senses the prior move through its own VPN/KC/MBON simulation, routes by VPN spike counts, plays its own action, and updates its cue head from its own game reward before imitation. The message is an action preference, not teacher weights, the hidden strategy, or a supplied correct-action label. It was recorded from an archived teacher that had learned Strategy B through its own rewards; the teacher was frozen during this pilot.

| Component | Used in this implementation | What this pilot establishes |
| --- | --- | --- |
| Real-wiring neural simulation | VPN stimulation and propagated spiking generate both flies' MBON rates; KC→MBON edges are present but their plastic multipliers remain one. | Neural features are used. Necessity of particular cells or edges is not isolated. |
| Teacher contextual bandit | External Q computation uses three standardized MBON rates; argmax gives one of three preferences. | An engineered controller extracts the preference from simulated activity. |
| DNge175 3141/3165 | Direct 300 Hz drive at 240–280 ms gives three codes `(1,0)/(0,1)/(1,1)`; actual somatic spikes are counted. | The count-based neural transmission is reliable under these fixed conditions. |
| Count buffer and decoder | A separate software buffer retains the two DN counts and decodes one action at 280 ms. | No persistent biological memory or downstream DN synaptic readout is demonstrated. |
| Student cue-head update | External own-reward update at 620 ms uses the cached premessage MBON features and action; an auxiliary softmax imitation step uses only the decoded DN preference. | Learning occurs in a hybrid readout. Neural KC→MBON dopamine plasticity is frozen. |
| Detailed DN output connectivity | DNs have outgoing connectome edges, but the update reads their counted spikes directly; the premessage features were already cached and student state resets next round. | No validated causal role for those output edges in the observed learning benefit. A DN-output lesion would not cleanly test this count-based route, so none was run. |

## Locked comparison

The two fixed archived Strategy B teachers were 11013/11014; fresh student seeds were 11017/11018. Each student had four matched 12×24 training arms: measured DN spikes, direct digital delivery of the *same frozen teacher preference at the same update time*, silence, and a fixed global permutation of the teacher preference tape carried through the same DN pathway. Student initial parameters, neural weights, opponent streams, keyed random seeds, 680 ms schedule, reward and imitation equations, and 0/6/12/24/48/96/288 frozen disconnected evaluations matched within each pair. Digital delivery omitted DN stimulation but preserved the ordinary sensory simulation. The scramble preserved the action histogram and 276 message opportunities; it moved 187/276 and 173/276 positions and coincided with the winning action on 91/276 and 100/276 observed rounds. It used neither cue nor outcome labels to generate the permutation.

Both archived teachers passed their earlier frozen learning gate (67/72 and 69/72). On the new training streams, their frozen preferences won 260/276 and 267/276 observed rounds, at least 70% for every cue. Each fresh student passed all 144 no-message/forced-code receiver probes: 12/12 DN decoding and VPN cue routing per cue and condition, with premessage MBON features exactly matching no-message probes. No stimulus or controller change was made after these checks.

### Frozen disconnected wins/72

| Student | Arm | 0 | 6 | 12 | 24 | 48 | 96 | 288 | Early mean 6/12/24/48 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11017 | Neural | 24 | 70 | 70 | 70 | 70 | 70 | 70 | 70 |
| 11017 | Digital | 24 | 70 | 70 | 70 | 70 | 70 | 70 | 70 |
| 11017 | Silent | 24 | 24 | 24 | 47 | 47 | 70 | 70 | 35.5 |
| 11017 | Scrambled neural | 24 | 24 | 47 | 70 | 70 | 70 | 68 | 52.75 |
| 11018 | Neural | 24 | 63 | 71 | 71 | 71 | 70 | 70 | 69 |
| 11018 | Digital | 24 | 63 | 71 | 71 | 71 | 70 | 70 | 69 |
| 11018 | Silent | 24 | 19 | 21 | 24 | 71 | 70 | 70 | 33.75 |
| 11018 | Scrambled neural | 24 | 63 | 58 | 64 | 71 | 70 | 70 | 64 |

Both students met the predeclared teacher-information effect versus **both** silence and scramble: neural early-mean advantages were **+34.5/+17.25 wins** for 11017 and **+35.25/+5** for 11018. Neural versus digital differences were **0** at every checkpoint and in the final policy; every paired training action, reward, reward update, imitation count, feature and Q score, final controller parameter, and frozen evaluation action matched exactly. The operational ±2-win similarity band passed. This is an observed equality under this code and schedule, not a general statistical equivalence claim.

Across students, trained neural delivery decoded **552/552** observed preferences; scrambled neural delivery also decoded **552/552**; digital delivery had **zero DN spikes** and supplied the same 552 trained preferences directly. All frozen evaluations had zero DN counts, no decoded message, exploration disabled, no reward or imitation update, unchanged controller weights, and intact KC→MBON multipliers. Trial-level training rows and all 72 frozen rows at every checkpoint are preserved in the write-once result (SHA-256 `78DDB6450A2E3B7A134C6A39ECBBF09AB77A8F5EFE97A52C27558B0852422068`). The two prior Strategy B artifacts retain their documented hashes and results unchanged.

## Interpretation and next step

The teacher's learned preference is genuinely carried by **simulated DN spikes** in the neural arm, and losing cue alignment weakens early retained learning. The digital arm does equally well because the software decoder delivers the same preference without error. Here the neural channel is an effective information-transport substrate, while the student decision is made before DN stimulation and its cached features plus external reward/imitation equations determine the later policy. The result does not establish a benefit over digital messaging, a necessary downstream DN synapse, biological brain-to-brain teaching, or original dopamine-only transfer. No lesion was run: with a count-buffer receiver, an outgoing-edge lesion would not isolate message dependence, and a broad sensory lesion could simply destroy cue recognition.

For HackPrinceton, the single useful next step is to **label the existing research demonstration with this distinction**: real simulated spikes carry a teacher preference, while the retained learning advantage arises in an engineered hybrid readout; direct delivery of the same preference gave the same outcome in this pilot. No judge-facing code was changed in this study.

Reproduce only in a copy without the write-once result: `node tools/check_rps_connectome_ablation.mjs`. Focused verification: `node --check tools/check_rps_connectome_ablation.mjs` and a read-only recount of every saved training reward, decoded DN code, frozen row, checkpoint win total, matched initial state, and opponent sequence. No Git operations or changes to prior artifacts were made.
