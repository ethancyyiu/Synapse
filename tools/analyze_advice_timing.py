"""Locked analysis for docs/advice-timing-protocol.md (Study 2). Final seeds 300-314 only."""
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(300, 315)
BOOT = 2026111560
results = [json.loads((ROOT / f"data/advice_timing/seed_{s}.json").read_text()) for s in SEEDS]
complete = [r for r in results if r["status"] == "completed"]
ARMS = ["alone", "early36", "spaced36", "randomTime36", "requested", "requestedRandom", "requestedShuffled", "full46"]
invalid = [s for s, r in zip(SEEDS, results) if r["status"] == "completed" and (
    not r["teacherFrozen"] or not r["synapsesUnchanged"] or [p["rounds"] for p in r["points"]] != [0, 24, 48])]


def at(r, rounds):
    return next(p for p in r["points"] if p["rounds"] == rounds)


def ci(diffs):
    rng = random.Random(BOOT)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
    return {"mean": statistics.mean(diffs), "bootstrap95": [means[250], means[9750]], "perSeed": diffs}


def paired(a, b):
    return ci([at(r, 48)[a] - (max(at(r, 48)[x] for x in b) if isinstance(b, tuple) else at(r, 48)[b]) for r in complete]) if complete else None


early, random_time = paired("early36", "spaced36"), paired("randomTime36", "spaced36")
timing_shown = any(x and (x["bootstrap95"][0] > 0 or x["bootstrap95"][1] < 0) for x in (early, random_time))
req_msgs = [r["requestsSent"] for r in complete]
req_scores = [at(r, 48)["requested"] for r in complete]
vs_spaced, vs_content = paired("requested", "spaced36"), paired("requested", ("requestedRandom", "requestedShuffled"))
c1 = bool(req_msgs) and statistics.mean(req_msgs) < 36
c2 = bool(req_scores) and statistics.mean(req_scores) >= 58 and sum(x >= 58 for x in req_scores) >= 12
c3 = bool(vs_spaced) and vs_spaced["bootstrap95"][0] > -4
c4 = bool(vs_content) and vs_content["bootstrap95"][0] > 4
enough = len(complete) >= 10 and not invalid
if not enough:
    request_verdict = "INSUFFICIENT (teacher readiness or integrity)"
elif c1 and c2 and c3 and c4:
    request_verdict = "SUPPORTED"
elif c1 and c2 and c3:
    request_verdict = "SAVED MESSAGES, BUT CONTENT BENEFIT NOT SHOWN"
else:
    request_verdict = "NOT SUPPORTED"
summary = {
    "protocol": "docs/advice-timing-protocol.md",
    "tau": complete[0]["tau"] if complete else None,
    "completed": len(complete),
    "teacherReadinessFailures": [{"seed": s, "teacherWins": r["teacherWins"]} for s, r in zip(SEEDS, results) if r["status"] != "completed"],
    "invalidSeeds": invalid,
    "meanWins": {str(t): {a: statistics.mean(at(r, t)[a] for r in complete) for a in ARMS} for t in (24, 48)} if complete else None,
    "recoveredAt48": {a: sum(at(r, 48)[a] >= 58 for r in complete) for a in ARMS},
    "messagesUsed": {"requestedMean": statistics.mean(req_msgs) if req_msgs else None, "requestedPerSeed": req_msgs},
    "timing": {"early36_minus_spaced36": early, "randomTime36_minus_spaced36": random_time,
               "verdict": "timing difference shown" if timing_shown else "no detectable timing difference"},
    "request": {"requested_minus_spaced36": vs_spaced, "requested_minus_strongerYokedControl": vs_content,
                "criteria": {"fewerMessages": c1, "recovers": c2, "notWorseThanSpaced": c3, "contentMatters": c4},
                "verdict": request_verdict},
    "reported": {"full46_minus_spaced36": paired("full46", "spaced36"), "requested_minus_alone": paired("requested", "alone")},
}
with (ROOT / "data/advice_timing_summary.json").open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({k: summary[k] for k in ("tau", "completed", "teacherReadinessFailures", "meanWins", "recoveredAt48", "messagesUsed")}, indent=1))
print("timing:", summary["timing"]["verdict"], {k: (round(v["mean"], 2), [round(x, 2) for x in v["bootstrap95"]]) for k, v in summary["timing"].items() if isinstance(v, dict)})
print("request:", request_verdict, summary["request"]["criteria"])
