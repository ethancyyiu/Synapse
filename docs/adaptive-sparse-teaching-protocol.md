# Prospective student-requested sparse RPS teaching protocol

Locked  **before adaptive neural training or held-out evaluation**. This is a new timing hypothesis. The prior fixed 25%/10% study remains failed on independent confirmation; its gates and artifacts are unchanged. The unchanged validated Strategy B action-preference teacher, student reward-plus-imitation controller, DN encoding, and frozen evaluation are reused.

## Feasibility and hypothesis

At 240 ms, the student cue head has already been selected from its own VPN spikes and its provisional RPS action and three Q scores have been computed, but the current opponent move and reward are not known until 280 ms. The Q-score gap exists but would evolve differently in each advice arm, confounding matched message exposure. The opt-in runner will instead count the student's **prior reward observations for its currently routed cue**. This is a simple experience-count proxy for uncertainty, not a calibrated probability or new learned model. Counts are maintained outside the readout for request scheduling only; they do not enter action selection, reward learning, imitation, or frozen evaluation. Because Strategy B's cue sequence is action-independent and receiver routing must pass, this produces the same request slots in every adaptive arm without sharing another arm's outcomes.

**Hypothesis:** concentrating a hard 25% teacher-message budget on early, less-experienced cue states will improve frozen disconnected early learning more than the prior evenly spaced 25% schedule, while trained content beats the same-timing shuffled, untrained, generic, and silent controls. The test may fail if generic early advice is equally sufficient.

Fly A cannot see private student counts. At each of the 276 observed-cue training rounds, the student sends **one digital request bit** to an opt-in teacher-side transmitter/scheduler at 240 ms. This is a modeled reverse channel outside the connectome; it conveys only request/no-request, no cue, Q values, weights, reward, correct action, or opponent outcome. A `1` permits that round's prerecorded frozen Fly A preference to be stimulated in Fly B's DNge175 neurons 3141/3165 during 240–280 ms. A `0` causes no DN stimulation. The receiver's existing measured-spike decoder alone supplies the student's imitation target. This implementation tests student-requested downlink timing; it does **not** establish a biological or all-neural bidirectional request pathway. Count **276 upstream request bits** plus actual DN downlink messages separately for each adaptive arm. The fixed and silent arms send no request bits. No overall bandwidth, energy, or cost saving is claimed.

## Fixed task, seeds, and rule

Deterministic observable **Strategy B**: `nextOpponent=(previousOpponent+2)%3`. Fly A trains from its own rewards for 12×24 rounds; Fly B trains for 12×24 rounds. Opponent seed and keyed random-stream conventions, initial student heads, feature normalization, epsilon 0.20, own-reward alpha 0.03, decoded-action imitation alpha 0.03, frozen KC→MBON weights, 680 ms round schedule, and three-action DN codes remain exactly as in `tools/check_rps_action_preference_strategy_b.mjs`. No teacher or student receives the hidden correct action as an input.

| Stage | Teacher | Student | Training opponent seeds | Held-out evaluation seeds |
| --- | ---: | ---: | --- | --- |
| Initial | 18101 | 18102 | 85000–85011 | 95000, 95001, 95003 |
| Conditional confirmation | 18103 | 18104 | 85012–85023 | 95004, 95006, 95009 |

These teacher/student seeds were absent from the existing RPS code and saved-result seed fields when locked. No replacement seeds are allowed. Each frozen checkpoint is a separate, disconnected 72-round evaluation at **0, 6, 12, 24, 48, 96, 288** student training rounds; exploration, reward learning, imitation, and messages are disabled. The primary metric is mean frozen wins/72 at rounds **6, 12, 24, 48**. Time to proficiency is the first sampled checkpoint reaching **58/72**; unreached is `null`. Final cue-specific wins and three action counts are recorded.

At **240 ms**, after the student selects its cue head and provisional action but **before** the 280 ms opponent/reward event, send request bit `1` iff (a) a real cue head 0–2 is routed, (b) that student's count of prior completed reward observations for that head is **less than 23**, and (c) fewer than **69** teacher messages have already been sent in this arm. Otherwise send `0` on an observed-cue round. No request is sent on each episode's cue-free first round. Increment that head's experience count only after the current round completes. Ties need no resolution; each head has its own integer count. The threshold **23 = 69/3** allocates at most 23 early advice opportunities to each of three cues without looking at rewards or the hidden next move. The cap is hard: at most **69/276** DN teacher messages. If fewer than 69 requests occur, leave the rest unused and report it. The transmitter does not fill silence automatically.

## Six matched arms and controls

1. **Adaptive trained:** frozen, own-reward-trained Fly A preference on requested rounds, decoded from measured DN spikes.
2. **Fixed trained:** same frozen teacher source and **69/276** send slots at zero-based observed-cue ordinals `j % 4 === 0`, as in the previous fixed study; no upstream request bits.
3. **Adaptive shuffled:** Fisher–Yates permutation of the trained preferences on the 69 adaptive slots only, using `rng(2026110921 + stageIndex)`, preserving their action histogram; actual DN delivery.
4. **Adaptive untrained:** same frozen neural source inference with untrained Fly A readout weights on the adaptive slots; actual DN delivery.
5. **Adaptive generic:** cue-independent random action preferences from `rng(2026110923 + stageIndex)` on adaptive slots; actual DN delivery.
6. **Alone:** no teacher stimulus or imitation, with the ordinary own-reward training; no upstream request bits.

Every arm uses the same student seed, initial neural/controller weights, 288 cue/opponent rounds, reward opportunities, keyed random streams, and held-out evaluation streams within a stage. The four adaptive arms independently compute their own request bits using the same local rule; equality of their request sequences is a required validity check, not imposed by copying another arm's outcomes. The scheduled shuffle is constructed from public cue occurrence counts and frozen teacher preferences before control training. It never uses the winning action or current reward. The generic generator never reads cue or outcome. Source/control correctness rates are measured only for analysis, not used to select seeds or regenerate controls. The fixed comparator has the same 69 downlink messages overall but different cumulative exposure by early checkpoint. Report cumulative messages at every checkpoint so a timing advantage is not mistaken for improved content per message.

## Prerequisites, correctness, and stopping

Before the six-arm comparison, require the new teacher to score **≥58/72**, **≥70% per observed cue**, and **≥12 selections of each action** in frozen evaluation after own-reward training, with KC→MBON weights frozen. Its source preference tape must select the winning action **≥70% per cue** and include all three actions. The new receiver must pass the unchanged 144-probe forced-code/no-message check: 12/12 decoded and routed per cue/condition, exact matched premessage MBON rates and features, intended DN spikes, and unchanged weights. The Alone student must start **≤50/72** and finish **≥58/72**, **≥70% per cue**, and **≥12 choices of each action**. If a prerequisite fails, save that result and stop before advice arms; do not replace a seed.

Before student comparison, verify the cue-only schedule construction requests at most 23 times per cue and 69 total, the fixed comparator uses 69 slots, all adaptive controls use the same candidate slots, and the shuffled scheduled multiset matches trained. During training assert: request decisions occur at 240 ms with current reward still unset; the policy uses only the routed cue and prior observation counts; all four adaptive request traces match; no advice is sent without a request; every sent target decodes from actual DN counts; skipped rounds have zero DN counts, decoded `-1`, and zero imitation while own-reward learning continues; no teacher label enters `imitate`; KC→MBON weights remain frozen; initial student weights and cue/opponent streams match. Frozen checkpoints must have no messages, requests, exploration, imitation, or learning, and must not change weights. Any violation invalidates the stage and stops it.

## Locked success gate

The adaptive trained arm passes a stage only if **all** hold:

1. Its early mean frozen wins exceed **each of the five other arms** (including fixed trained, adaptive shuffled/untrained/generic, and Alone) by at least **4/72**.
2. It first reaches **58/72 by round 48**, strictly earlier than each other arm; an arm never reaching 58 is later by definition.
3. Its final disconnected result reaches **≥58/72**, **≥70% wins for every observed cue**, and **≥12 selections of each RPS action** on the 69 observed held-out rounds.
4. All prerequisites, message-budget and transmission checks, matched-control checks, and frozen-weight checks pass.

If the initial stage fails any gate, preserve the failure and stop; do not run confirmation. Only a full initial pass triggers the fixed independent confirmation seeds and **unchanged** rule/threshold/gates. A teacher-specific timing claim requires both stages to pass. The prior fixed sparse failure remains valid regardless of this result. The direct-digital ablation found identical learning and limits any neural superiority claim. No further seeds, schedule variants, threshold changes, or tuning in this study.

Reproduce only in a copy without the write-once output: `node tools/check_rps_adaptive_sparse_teaching.mjs initial`; if and only if it passes, `node tools/check_rps_adaptive_sparse_teaching.mjs confirmation`.
