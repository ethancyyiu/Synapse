# DNge175 teacher-message audit

## Scope

This note is a read-only audit of the archived DNge175 teacher-guided RPS evidence. It does not change the simulator, learning rule, demo, or saved experiment outputs.

## Executive conclusion

Fly A does communicate usable information to Fly B, but the archived evidence does not show that the learned DNge175 message adds a retained teacher-specific learning benefit beyond generic exploration. The message is real and physically decodable, but it is also compressed and weakly informative relative to the task.

The current evidence supports a narrow claim: the teacher signal can alter immediate action selection and create useful reward opportunities. It does not support a stronger claim that the signal provides uniquely useful retained learning over matched non-teacher or generic controls.

## What Fly A actually communicates

Fly A does not send a direct hidden strategy, a three-way target action, or a full value vector. In the archived teaching protocol, Fly A learns a cue-conditioned Q estimate from its own MBON features and uses that to evaluate Fly B's public provisional action. The teacher then emits a one-bit approval signal through the DNge175 interface.

In the code path, the teacher's reward-trained contextual-bandit computes a score for each candidate action from MBON-derived features, then compares the scored action against the two alternatives. The delivered signal is a yes/no decision:

- approve if the current proposal is scored above the alternatives
- reject otherwise

The DNge175 code path carries that bit using the actual measured spike counts from neurons 3141 and 3165, with the student decoding one bit from the measured counts. This is a communication of preference, not a direct action label.

## What information is learned versus engineered versus environmental

### Learned by Fly A

- Fly A's own MBON-driven Q estimates
- its preference for one candidate action over alternatives under the current cue
- the resulting approval or rejection decision for Fly B's public action proposal

### Engineered by the protocol

- the public provisional action exposed to Fly A
- the fixed one-bit DNge175 encoding
- the transform from measured DN spikes into a decoded bit
- the use of the decoded bit as a gate or veto on Fly B's provisional action
- the fixed cue-specific student readout and selected-head update logic

### Supplied by the environment

- previous-opponent cue information
- the round reward outcome
- the actual opponent move and noisiness in the task
- the stochastic or deterministic structure of the hidden strategy

The teacher signal is therefore a learned preference filtered through an engineered neural channel and then used by a student that also receives ordinary game rewards and cue observations from the environment.

## Message informativeness beyond generic exploration

The archived studies show that the trained teacher message is measurably informative in a limited sense:

- trained advice often yields better immediate action filtering
- it creates more rewarded action opportunities in the early rounds
- it tends to improve the student's immediate online behavior

This is evidence that the message is not random and that the teacher's learned preference is not entirely irrelevant.

However, that does not establish superiority over generic exploration. The archived five-arm and matched-veto studies repeatedly show that:

- generic veto or shuffled advice can create similar early reward opportunities
- untrained or shuffled message controls often reach the same frozen disconnected performance
- trained advice does not clearly exceed the matched controls on retained learning

The strongest accurate statement is therefore:

The message carries a real, teacher-derived signal about candidate quality, but the archived data do not demonstrate that it contains task-specific information that generic exploration cannot reproduce.

## One-bit information loss

The current message compresses a three-way action-value judgment into one bit.

That loses several kinds of information:

- the preferred action identity
- the ranking among the alternatives
- the magnitude of the preference
- the confidence or certainty of the teacher signal
- the distinction between "slightly worse" and "much worse"

When the teacher rejects a proposed action, the message does not tell the student which of the other two actions was preferred. It only says the proposed action was not the best among the teacher's current scores.

This creates an ambiguity that matters for a three-action task such as RPS. The teacher's bit can accept or reject a candidate, but it cannot cleanly encode:

- which action should be preferred
- which action is second-best
- how strongly the teacher prefers that option

A different encoding could in principle carry:

- a preferred action
- a ranked three-way preference
- a confidence measure or strength signal

But those would require a richer communication representation than the current binary DNge175 code. The archived system is compatible with an approval/rejection channel, but not with a full action-ranking channel without redesign.

## Why superior teacher advice did not translate into better retained learning

The most coherent explanation from the saved records is that the teacher message helped early action selection but did not add uniquely useful retained information beyond what generic exploration already supplied.

The evidence indicates:

- trained advice improved immediate online play and produced more targeted updates
- the cue-specific heads can learn quickly once they get a few correct rewards
- generic or shuffled controls often supplied enough similar reward opportunities to converge to the same frozen disconnected policy
- the teacher-specific advantage gate failed repeatedly under locked comparisons

This is not a failure of message delivery. It is a failure of specific advantage: the student could get enough useful information from other exploration paths, so the trained teacher signal did not produce a uniquely better retained policy.

## Final assessment

### 1. What does Fly A actually communicate?
Fly A communicates a learned yes/no preference about Fly B's current provisional action, using its own MBON-driven value estimates and the DNge175 one-bit channel.

### 2. Is the message demonstrably informative beyond generic exploration?
Only at the immediate-action level. The archived results show useful filtering and early reward opportunities, but no convincing retained-learning gain beyond matched generic or untrained controls.

### 3. How much task information is lost in the one-bit encoding?
A substantial amount. The encoding loses preference magnitude, ranking, and confidence, and it collapses the three-action decision into a binary approval decision.

### 4. Why does better teacher advice not produce better retained learning?
Because the archived evidence supports fast saturation of the student's cue-head learning and matched generic exploration produces similar downstream policy gains. The teacher signal is not uniquely necessary for the retained policy.

### 5. What single change or investigation is justified next?
The current evidence does not justify changing the communication mechanism. The justified next step is a separate, explicitly prospective study of a more informative message encoding only if the project wants to continue the communication hypothesis. Under the current evidence, the more defensible choice is to stop the teacher-guided research branch and preserve the negative result as a valid finding.

## Decision

The archived evidence supports the conclusion that the current DNge175 one-bit teacher signal is real but insufficiently informative to justify continued optimization of the same mechanism. No new simulator runs, no new training, and no channel change are justified by the available evidence.
