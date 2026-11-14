# Locked rule-switch teaching pilot

Locked , before any run. Earlier results show teaching speeds early learning, but students alone catch up. This pilot asks one question: after the opponent's rule changes, does a teacher who already knows the new rule help Fly B adapt within a short fixed window? It uses at most three seeds and changes one thing, the opponent's rule partway through.

## Design

For seed `s` in 0, 1, 2: teacher `24001 + 2s`, student `24002 + 2s`.

1. **Teacher.** Trains 12×24 rounds on Strategy B from its own rewards (opponent seeds `84000 + 100s + e`), then freezes. Gate: at least 58/72 frozen wins on its check.
2. **Shared history.** One student trains 96 rounds on Strategy A with no messages (opponent seeds `85000 + 100s + e`). Its weights are copied into three students, so all arms start the switch identical.
3. **Switch.** The opponent becomes Strategy B. For 48 rounds (opponent seeds `86000 + 100s + e`) three arms train with the existing action-preference round: `trained` (Fly A's preference), `random` (one uniform random action on each cued round, same exposure), and `alone` (no message).
4. **Frozen checks.** 72 rounds on Strategy B (opponent seeds `96000 + 10s + e`), teacher disconnected, with no learning or exploration, at 0, 6, 12, 24, 48 rounds after the switch.

## Fixed criterion

For each seed, `advantage = wins(trained) - max(wins(alone), wins(random))` at round 48 after the switch.

**Pilot pass:** the teacher gate passes for all three seeds, and the advantage is at least +4/72 on at least two of three seeds. Otherwise the pilot fails. All seeds are reported. No seed is replaced, and the window, thresholds, and code are not changed after seeing results.

## Limits

This is a three-seed pilot on fixed RPS rules. A pass would justify a locked confirmation only. It measures adaptation within a fixed window, so a student alone may still catch up given more rounds, and shuffled and untrained controls are not included.
