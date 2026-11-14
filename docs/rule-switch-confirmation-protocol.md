# Locked rule-switch confirmation

Locked , before any run. The rule-switch pilot (`docs/rule-switch-pilot-protocol.md`) passed 3/3 but had no shuffled or untrained-teacher arm and a 48-round window. This confirmation reruns the same design on new seeds, adds those two arms, and extends the window to see whether students alone recover. It uses three seeds and changes nothing else: same teacher training, shared Strategy A history, `preferenceRound`, dopamine rule, and frozen checks.

## Design

For seed `s` in 0, 1, 2: teacher `25001 + 2s`, student `25002 + 2s`.

1. **Teacher.** Trains 12×24 rounds on Strategy B (opponent seeds `87000 + 100s + e`), then freezes. Gate: ≥58/72 frozen wins (seeds `97000 + 10s + e`).
2. **Shared history.** One student trains 96 rounds on Strategy A with no messages (seeds `88000 + 100s + e`), then is copied into every arm.
3. **Tapes.** The frozen teacher's preference on each switch round is recorded in a dry run of the same rounds (opponent cues do not depend on the student's moves). The real run must reproduce the tape exactly.
4. **Switch.** Strategy B for 144 rounds (seeds `89000 + 100s + e`), five arms:
   - `trained`: Fly A's live preference.
   - `random`: one uniform random action per cued round.
   - `untrained`: the preference of the same teacher network with untrained readout weights.
   - `shuffled`: the trained tape shuffled across all cued rounds (rng `2026110904 + 2s`), so action frequencies match but timing is broken.
   - `alone`: no message.
   All arms use `preferenceRound`. Each guided arm runs next to its own copy of `alone`, and all `alone` copies must stay identical.
5. **Frozen checks.** 72 rounds on Strategy B (seeds `97000 + 10s + e`), teacher disconnected, at 0, 24, 48, 96, and 144 rounds after the switch.

## Fixed criterion

`advantage = wins(trained) - max(wins(alone), wins(random), wins(untrained), wins(shuffled))` at 48 rounds after the switch.

**Pass:** the teacher gate passes for all three seeds, and the advantage is ≥ +4/72 on at least two of three seeds. All seeds are reported. No seed is replaced, and nothing is changed after seeing results.

**Descriptive only:** wins at 96 and 144 rounds, and the first checkpoint at which `alone` comes within 4 wins of `trained`.

## Limits

Three seeds on fixed RPS rules. A pass supports "faster adaptation after a rule change" under these controls, not better final performance or real-world transfer.
