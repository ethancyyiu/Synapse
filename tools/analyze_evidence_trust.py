"""Locked analysis for docs/evidence-trust-protocol.md (Study 3). Final seeds 30-44 only."""
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(30, 45)
BOOT_SEED = 2026111530
results = [json.loads((ROOT / f"data/evidence_trust/seed_{s}.json").read_text()) for s in SEEDS]
complete = [r for r in results if r["status"] == "completed"]
invalid = [s for s, r in zip(SEEDS, results) if r["status"] == "completed" and (
    not r["teachersFrozen"] or not r["synapsesUnchanged"] or [p["rounds"] for p in r["points"]] != [0, 24, 48, 96])]
ARMS = ["alone", "trained", "stale", "evidenceStale", "evidenceTrained", "discountStale", "discountTrained"]


def at(r, rounds):
    return next(p for p in r["points"] if p["rounds"] == rounds)


def paired(a, b, rounds):
    diffs = [at(r, rounds)[a] - at(r, rounds)[b] for r in complete]
    if not diffs:
        return None
    rng = random.Random(BOOT_SEED)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {"mean": statistics.mean(diffs), "bootstrap95": [means[250], means[9750]], "perSeed": diffs}


protection = paired("evidenceStale", "stale", 96)
no_harm = paired("evidenceStale", "alone", 96)
kept = paired("evidenceTrained", "trained", 48)
req1 = bool(protection and no_harm and protection["bootstrap95"][0] > 4 and no_harm["bootstrap95"][0] > -4)
req2 = bool(kept and kept["bootstrap95"][0] > -4)
enough = len(complete) >= 10 and not invalid
verdict = "FULL PASS" if enough and req1 and req2 else "PARTIAL" if enough and (req1 != req2) else "FAIL"

summary = {
    "protocol": "docs/evidence-trust-protocol.md",
    "finalSeeds": list(SEEDS),
    "completed": len(complete),
    "teacherReadinessFailures": [{"seed": s, "status": r["status"], "teacherWins": r.get("teacherWins"),
                                  "staleTeacherWins": r.get("staleTeacherWins")}
                                 for s, r in zip(SEEDS, results) if r["status"] != "completed"],
    "invalidSeeds": invalid,
    "meanWins": {str(t): {a: statistics.mean(at(r, t)[a] for r in complete) for a in ARMS} for t in (0, 24, 48, 96)} if complete else None,
    "meanTrust": {str(t): {k: statistics.mean(at(r, t)["trust"][k] for r in complete) for k in ("evidenceStale", "evidenceTrained")}
                  for t in (0, 24, 48, 96)} if complete else None,
    "requirement1": {"protection_evidenceStale_minus_stale_96": protection, "noHarm_evidenceStale_minus_alone_96": no_harm, "holds": req1},
    "requirement2": {"benefitKept_evidenceTrained_minus_trained_48": kept, "holds": req2},
    "reported": {
        "evidenceStale_minus_discountStale_96": paired("evidenceStale", "discountStale", 96),
        "evidenceTrained_minus_discountTrained_48": paired("evidenceTrained", "discountTrained", 48),
        "discountStale_minus_alone_96": paired("discountStale", "alone", 96),
        "discountTrained_minus_trained_48": paired("discountTrained", "trained", 48),
        "stale_minus_alone_96": paired("stale", "alone", 96),
        "trained_minus_alone_48": paired("trained", "alone", 48),
    },
    "verdict": verdict,
}
output = ROOT / "data/evidence_trust_summary.json"
with output.open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({k: summary[k] for k in ("completed", "teacherReadinessFailures", "invalidSeeds", "meanWins", "requirement1", "requirement2", "verdict")}, indent=1))
