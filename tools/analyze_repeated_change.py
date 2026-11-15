"""Locked analysis for docs/repeated-change-protocol.md (Study 4). Final seeds 0-14 only."""
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(0, 15)
BOOT_SEED = 2026111540
results = [json.loads((ROOT / f"data/repeated_change/seed_{s}.json").read_text()) for s in SEEDS]
complete = [r for r in results if r["status"] == "completed"]
invalid = [s for s, r in zip(SEEDS, results) if r["status"] == "completed" and (
    not r["teachersFrozen"] or not r["synapsesUnchanged"] or [p["rounds"] for p in r["points"]] != [0, 48, 72, 96])]
ARMS = ["alone", "trained", "random", "shuffled"]


def at(r, rounds):
    return next(p for p in r["points"] if p["rounds"] == rounds)


def advantage(rounds):
    diffs = [at(r, rounds)["trained"] - max(at(r, rounds)[a] for a in ("alone", "random", "shuffled")) for r in complete]
    if not diffs:
        return None
    rng = random.Random(BOOT_SEED)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {"mean": statistics.mean(diffs), "bootstrap95": [means[250], means[9750]], "perSeed": diffs,
            "seedsAtMost4": sum(d <= 4 for d in diffs)}


primary = advantage(96)
if len(complete) < 10:
    verdict = "STOPPED AT TEACHER READINESS"
elif invalid:
    verdict = "INVALID"
else:
    verdict = "PASS" if primary["bootstrap95"][0] > 4 else "FAIL"
summary = {
    "protocol": "docs/repeated-change-protocol.md",
    "finalSeeds": list(SEEDS),
    "completed": len(complete),
    "teacherReadinessFailures": [{"seed": s, "status": r["status"], "teacherBWins": r.get("teacherBWins"),
                                  "teacherAWins": r.get("teacherAWins")} for s, r in zip(SEEDS, results) if r["status"] != "completed"],
    "invalidSeeds": invalid,
    "meanWins": {str(t): {a: statistics.mean(at(r, t)[a] for r in complete) for a in ARMS} for t in (0, 48, 72, 96)} if complete else None,
    "primary_advantage96": primary,
    "reported": {"advantage48_afterChange1": advantage(48), "advantage72": advantage(72)},
    "verdict": verdict,
}
output = ROOT / "data/repeated_change_summary.json"
with output.open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({k: summary[k] for k in ("completed", "teacherReadinessFailures", "invalidSeeds", "meanWins", "primary_advantage96", "verdict")}, indent=1))
