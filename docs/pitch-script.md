# Synapse: 2-minute pitch and judge Q&A

## Before judges arrive
- Run `python tools/serve.py`, open `http://localhost:8766/`, and stay at the top of the **Simulation** tab.
- Leave the CPU idle: close other simulations, and keep the tab in front.
- Do not press anything yet. **▶ Watch the demo** starts the 1½-minute recorded run. The page scrolls by itself to whichever brain is active, so don't scroll during the run (scrolling turns the auto-follow off).

## Script (about 2 minutes)

**Hook, 15 s.** *(Top of page.)*
"When AI systems from different companies need to cooperate, they hit a wall: incompatible architectures, privacy constraints, no way to share what they've learned. We built a proof-of-concept that breaks through that wall with 12 bytes."

**Bridge to demo, 5 s.**
"We used the most detailed neural substrate we have — the real fruit-fly connectome, 6,900 neurons — as a controlled test bed. Watch."

**Demo, 55 s.** *(Press ▶ Watch the demo.)*
- "Fly A learns Rock-Paper-Scissors from its own wins and losses. Think of it as one AI system adapting to a new environment."
- *(Fly A's brain is on screen, with its round count and then its test score, 67 of 72.)*
- "Now the opponent changes strategy — the world shifted. Fly A has already adapted. Fly B hasn't. Fly A sends its current best guess as spikes on two neurons. That's the 12-byte message. Fly B's architecture is different — there's nothing to copy, no weight transfer possible. It just gets the hint."
- "Three Fly Bs: one taught, one getting random noise, one learning alone. Green squares are wins."
- "We unplug Fly A completely, then test Fly B on its own." *(Scoreboard: taught 71, alone 32, random 57, out of 72.)*

**Result, 30 s.** *(Click the **Stats** tab on the right edge, then point at the 15-run chart.)*
"This was one recorded run, one of the stronger ones. We repeated the hardest case 15 times, with fresh brains and locked rules decided before running. After the opponent changed strategy, taught flies won 68 of 72 games; flies learning alone won 10, and the strongest control, shuffled messages, 33. Against the best control in each run the lead is 25. The teacher beat random, shuffled, and untrained-teacher messages in 14 of 15 runs. And the speed-up is the point: within 48 rounds, the taught fly reached 58 of 72 wins in all 15 runs; the fly learning alone, in none." *(Point at the "How much faster?" bars.)*

**Real-world stakes, 15 s.** *(Click the Research tab.)*
"Think about what 48-versus-6 decisions means at scale. A network intrusion detector running 1,000 decisions per second: 48 seconds of blind exposure versus 6. Two hospitals with different model architectures that can't share patient data — one has seen the new strain, the other hasn't. A fleet of robots from two manufacturers, one has adapted to the new layout. In every case, the message is 12 bytes. Fly B's learner has a different structure than Fly A's — weights couldn't be copied even if we tried. The hint still works. That is what this result shows."

## Three answers to have ready (15 seconds each)

- **"Do the spikes matter?"** "No. A plain digital message works exactly as well, and we tested that. What we show is that learned advice survives a neural channel. Spikes would matter on brain-like chips, which only talk in spikes."
- **"What did you build, and what did you start from?"** "We started from an open-source fly-brain simulator that plays a rhythm game. Two flies teaching each other, the message channel, every experiment and control, and this page are ours. We also wrote our own version of the simulator and the main result held on it in the 10 of 15 runs where the teacher learned."
- **"Does it work outside this game?"** "Yes — on a fault-response task where a machine raises alarms in random order and one alarm's correct response changes. 15 runs. Taught fly led the best control by 19 wins on average (95% CI 11.8 to 26.2). That passed. With a small change, where recovery alone was quick, teaching added nothing — which is what you'd expect and want. Two pilots before the full cohort had failed, and we show those too."

## Likely questions

| Question | Answer |
|---|---|
| Do the spikes matter, or would sending a number work? | A direct digital message worked equally well; that's in our ablation. We show learned knowledge survives a real neural channel, not that it beats digital. |
| Is the fly brain actually learning? | It's a hybrid. The connectome senses and carries the message, a small learned readout picks the move, and Fly B's imitation update is engineered. We never changed the connectome or the dopamine rule. |
| Is this one lucky run? | No. There's a 20-pair study (+10.3 wins, 95% CI +7.3 to +13.6) and a 15-run rule-change study (+24.9, 95% CI +17.8 to +32.1). All protocols were locked before running, with failures reported. |
| Why not just copy the weights? | Fly B's learner has a different structure (48 numbers vs 13), so there is nothing to copy into. Advice works across different designs. |
| Doesn't the alone fly catch up? | Yes, eventually. The benefit is speed after a change, and that's what matters when the world shifts. |
| Real-world use? | The Research tab lists seven, each with a published study behind the problem: mixed robot fleets, hospitals, banks and fraud, underwater vehicles, brain-like chips, brain-computer interfaces, and old equipment that cannot be updated. Brain-like chips are the one case where spikes themselves would matter, because that hardware talks in spikes. We tested a second task, a fault-response task with alarms in random order. After a large change the taught fly led the best control by 19 wins on average over 15 runs (95% CI 11.8 to 26.2), which passed. With a small change the fly recovered alone within 24 rounds, so teaching added nothing, and two 3-run pilots before the full test had failed their rules. Not tested outside simulation: our practical package-recovery task stopped because the teacher didn't learn it well enough. A sensor-drift benchmark is the next step. |
| What if the teacher is wrong? | We tested it. A teacher trained on the old rule held Fly B 30 wins below learning alone. We then let Fly B learn from its own wins how much to trust the teacher: that got back 34 of those wins and kept the good teacher's speed-up, but it still ended 7.5 wins below learning alone, so that study failed its own pass rule and we report it as failed. |
| What if it hears a good teacher and a bad one? | We tested that too. Fly B heard both every round and was not told which was which. Keeping a trust score for each, it finished at 70 of 72, exactly what it gets from the good teacher alone, in all 14 runs. The honest catch: a fly that simply followed both got to 62 by the end as well, so the benefit of choosing was +8.4 wins with a lower bound of 3.8, just under the 4 we had set in advance. So that study failed its pass rule narrowly, and we say so. The benefit of choosing shows up earlier: 68 against 50 at 48 rounds. |
| Random messages beat learning alone too. Why? | Yes: 32 against 10. Any message loosens the old habit after the rule changes. That is why our headline is the +24.9 lead over the best control, not the gap to learning alone. |
| This is just RPS — what's the real-world connection? | RPS is a proxy for any cue-response task where the mapping changes and two agents can't share internals. The structure is identical to: a network intrusion detector that sees a new attack pattern and needs to tell a differently-built partner (12 bytes, not a model dump); two hospitals with different architectures that can't share patient data but one has seen the new strain; a robot fleet where one unit adapted to a new obstacle layout and needs to inform the others. We didn't build those systems — we proved the mechanism works under rigorous controls. That's what a first result is. |
| Why use a fruit fly? | The fruit fly connectome is the most completely mapped neural circuit we have — every neuron, every synapse, real data from Janelia Research Campus. It gives us a reproducible, biologically grounded substrate to test the mechanism. The lesson isn't about flies; it's about whether constrained neural-style channels can carry useful learned information between differently-architected agents. |
| Does it work with a noisy opponent? | No, and we report that. With an opponent that follows its rule 80% of the time, the locked test required the teacher to win for every student; only one of the students passed, so the test failed. |
| What if spikes are lost? | We dropped 10% and 30% of the message spikes. For one student teaching still beat the controls; for the other, shuffled advice did equally well (both 71/72), so the all-student check failed. Honest answer: robustness is partial. |
| Why does random messaging get 57 in the recorded run? | That recording is one of the stronger runs, and random advice still nudges exploration. Across all 15 runs, random messages averaged 32 against the teacher's 68, and matched the teacher in only 1 of 15. |
| Isn't this just distillation? | The idea is close to policy distillation: Fly A shares its preferred actions, not weights. What's new is doing it through a real connectome's spikes, between differently built learners, with locked controls. |
| Real-world use? | Robots or drones of different makes, privacy-sensitive organisations (advice instead of patient data or weights), and low-bandwidth devices. Not tested outside simulation: our practical package-recovery task stopped because the teacher didn't learn it well enough. A sensor-drift benchmark is the next step. |

## Fallbacks
- If the recording stalls, reload and press ▶ Watch the demo again. It's display only, so nothing is lost.
- If there's no time for the demo, show the Result chart and say the headline.
