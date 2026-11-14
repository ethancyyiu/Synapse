# Synapse: 2-minute pitch and judge Q&A

## Before judges arrive
- Run `python tools/serve.py`, open `http://localhost:8766/`, and stay at the top of the **Simulation** tab.
- Leave the CPU idle: close other simulations, and keep the tab in front.
- Do not press anything yet. **▶ Watch the 90-second story** starts the recorded rule-change run. The page scrolls by itself to whichever brain is active, so don't scroll during the run (scrolling turns the auto-follow off).

## Script (about 2 minutes)

**Question, 15 s.** *(Top of page.)*
"When the rules change, can one learner help another adapt faster? We tested that with two separate simulated fly brains."

**Demo, 90 s.** *(Press **Watch the 90-second story**.)*
- "Fly A learns from wins and losses. Then the opponent changes its rule."
- "Fly A sends the move it now prefers as spikes on two neurons. Fly B still plays and learns from its own results."
- "We disconnect Fly A and test Fly B alone. This recording is one example, so the study result matters more than its final score."

**Result, 25 s.** *(Click **Stats**. Point to the 48-round checkpoint in the 15-run chart.)*
"Across 15 independent runs, taught Fly Bs averaged 68 wins out of 72. The shuffled-message control averaged 33. Comparing each taught fly with its strongest control in the same run, the average lead was 25 wins. That is the result: faster adaptation after a rule change, measured after the teacher was disconnected."

**Boundary, 10 s.**
"This is a hybrid simulation with an engineered learning step. The same advice sent digitally worked equally well. We have a controlled learning-speed result, not biological fly-to-fly teaching."

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
- If the recording stalls, reload and press **▶ Watch the 90-second story** again. It's display only, so nothing is lost.
- If there's no time for the demo, show the Result chart and say the headline.
