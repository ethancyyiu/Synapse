# Prospective sparse DN action-preference teaching protocol

Locked  **before any sparse-result simulation**. This is a new communication-frequency study of the unchanged Strategy B post-sensory action-preference system. The existing Strategy A/B successes, digital ablation, and failed experiments remain separate.

## Hypothesis and feasibility

A reward-trained Fly A's three-action preference, decoded from actual DNge175 spikes on predetermined rounds, may improve Fly B's **disconnected frozen early learning** even when most training rounds are silent. On silent rounds the established `studentRound` already accepts `target=null`: it drives neither DN, decodes `-1`, performs no imitation update, and still runs the same 680 ms neural simulation and the ordinary own-reward controller update. Per-round neural and controller random streams are keyed independently of prior draws. The Strategy B opponent schedule depends on its previous move, not Fly B's action. Frozen evaluation has always used `target=null, learn=false`. Thus only whether an already computed teacher preference is transmitted changes; the controller, message code, learning equations, timing, and sensory pathway remain fixed.

The teacher preference tape is generated in advance by frozen, own-reward-trained Fly A neural inference; no correct action or opponent outcome enters the message. Student imitation receives only the preference decoded from measured DN spikes. The teacher's action preference is an engineered, three-symbol spike code, not biological memory transfer.

## Fixed assignments and schedules

Primary task: deterministic observable **Strategy B**, `nextOpponent=(previousOpponent+2)%3`. Teacher trains with its own rewards for 12 episodes × 24 rounds; Fly B trains for 12 × 24 rounds. Cue-free first rounds remain in training, have no advice, and count toward checkpoints. Frozen checkpoints are **0, 6, 12, 24, 48, 96, 288** training rounds, each 72 disconnected rounds over three fixed opponent streams. The primary early metric is mean frozen wins out of 72 at **6, 12, 24, 48**. First checkpoint with at least **58/72** wins measures time to proficiency; `null` means unreached. Final frozen wins, per-cue wins, action counts, and all trial rows are retained.

| Stage | Teacher seed | Student seed | Training opponent seeds | Frozen opponent seeds |
| --- | ---: | ---: | --- | --- |
| Initial | 18001 | 18002 | 84000–84011 | 94000, 94001, 94003 |
| Confirmation, conditional | 18003 | 18004 | 84012–84023 | 94004, 94006, 94009 |

These seeds were checked against the existing code and documentation before execution and are not replacements for previously evaluated seeds. Existing per-round keys remain `studentSeed*10000000 + episode*1000 + round`; neural and action RNGs remain `rng(key+2)` and `rng(key+501)`.

Let `j` be the zero-based ordinal of **observed-cue** training rounds, ignoring each episode's cue-free first round. Send on:

| Nominal budget | Deterministic send rule | Expected messages of 276 eligible rounds |
| --- | --- | ---: |
| 100% | every `j` | 276 |
| 25% | `j % 4 === 0` | 69 |
| 10% | `j % 10 === 0` | 28 (10.14%; indivisible rounds) |

No outcome, teacher confidence, student action, or hidden mapping selects a send round. At every budget use five arms from the **same** initial student weights and matched cues, opponents, reward opportunities, keyed streams, and frozen evaluation: trained teacher, untrained teacher, shuffled trained preferences, cue-independent generic random preferences, and silence. The silent arm is shared across budgets. The shuffled arm permutes only preferences on scheduled sends with a fixed Fisher–Yates stream `rng(2026110900 + stageIndex*100 + budgetPercent)`, preserving that budget's exact action histogram and exposure. The generic arm draws 0–2 uniformly with `rng(2026110950 + stageIndex*100 + budgetPercent)` on those same scheduled rounds. Untrained Fly A uses the same frozen source-inference path with untrained controller weights. All four nonsilent arms physically stimulate the same DN targets; a silent slot sends no message. The generic control is a synthetic cue-independent preference, disclosed as such.

## Fixed validity and stop rules

Before student comparison, require: teacher frozen **≥58/72**, **≥70%** per cue, **≥12** of each action, KC→MBON frozen; source tape's trained preference wins **≥70% per cue**, contains all three actions, and teacher weights remain frozen; receiver's original matched-stream, forced-code check passes every cue/code (12/12 decoding and VPN routing, exact paired pre-pulse MBON features, no weight changes). The silent student must start **≤50/72** and finish **≥58/72**, **≥70%** per cue, and select each action **≥12** times. If any of these prerequisites fails, save the result and stop without sparse student arms. No replacement seed.

Before training, verify scheduled counts exactly 276/69/28, identical schedule for all messaged controls at each budget, no messages on cue-free rounds, shuffled scheduled multiset equality, and no change to weights from `imitate(null)`. During training require every silent round to decode `-1`, have zero DN spikes and zero imitation; every sent round must decode its scheduled target from actual measured DN spikes and have one imitation step; reward updates must occur on all eligible and silent rounds. Afterward require identical initial weights, cue/opponent sequences, and 7 disconnected frozen evaluations across all arms, with no teaching, exploration, readout learning, or KC→MBON weight changes during evaluation. Any violation invalidates the stage and stops it.

## Locked outcome gates

For **each budget independently**, the trained-message arm passes only if all of the following hold:

1. Its early mean frozen wins exceed **silence and each exposure-matched untrained, shuffled, and generic control by at least 4/72**.
2. It first reaches **58/72 by round 48**, strictly earlier than silence and all three controls at that budget; controls never reaching 58 also satisfy the ordering.
3. Its final disconnected evaluation reaches **≥58/72**, **≥70% wins for every observed cue**, and **≥12 selections of each action** over the 69 observed rounds.
4. All transmission and matched-run validity checks above pass.

Record every budget, including failures. A sparse budget (25% or 10%) passing the initial stage triggers **one independent confirmation** using the fixed confirmation assignments above and the same full three-budget/five-arm protocol. A sparse claim requires that **the same budget** pass both stages. A failure on confirmation remains a failure; no further seeds, rounds, schedule edits, or tuning. If neither sparse budget passes initially, stop without confirmation. The 100% arm is the reference, not a substitute for sparse success. Report actual message counts and learning effects, not bandwidth, energy, or real-world cost savings. The previous direct-digital ablation found identical learning behavior and precludes a claim of neural superiority.

Reproduction (only in a copy lacking its write-once output): `node tools/check_rps_sparse_neural_teaching.mjs initial`; if triggered, `node tools/check_rps_sparse_neural_teaching.mjs confirmation`.
