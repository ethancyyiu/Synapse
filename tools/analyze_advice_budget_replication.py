"""Locked analysis for docs/advice-budget-replication-protocol.md (Study 5). Final seeds 200-214 only.

Same bootstrap method as tools/analyze_advice_budget_followup.py, with a new fixed seed.
Writes data/advice_budget_replication_summary.json once.
"""
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(200, 215)
BOOT_SEED = 2026111520
results = [json.loads((ROOT / f"data/advice_budget_replication/seed_{s}.json").read_text()) for s in SEEDS]
complete = [r for r in results if r["status"] == "completed"]
EXPECTED = {"trained36": 36, "random36": 36, "shuffled36": 36, "trained46": 46}
invalid = [s for s, r in zip(SEEDS, results) if r["status"] == "completed" and (
    r["messages"] != EXPECTED or not r["teacherFrozen"] or not r["synapsesUnchanged"] or
    [p["rounds"] for p in r["points"]] != [0, 24, 48])]


def interval(differences):
    rng = random.Random(BOOT_SEED)
    means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10_000))
    return [means[250], means[9750]]


def at(r, rounds):
    return next(p for p in r["points"] if p["rounds"] == rounds)


def paired(rows, a, b):
    diffs = [row[a] - (max(row[x] for x in b) if isinstance(b, tuple) else row[b]) for row in rows]
    return {"mean": statistics.mean(diffs), "bootstrap95": interval(diffs)} if diffs else None


rows48 = [at(r, 48) for r in complete]
rows24 = [at(r, 24) for r in complete]
arms = ["alone", "trained36", "random36", "shuffled36", "trained46"]
recovered36 = sum(row["trained36"] >= 58 for row in rows48)  # failed-teacher seeds count as not recovered
mean36 = statistics.mean(row["trained36"] for row in rows48) if rows48 else None
specific = paired(rows48, "trained36", ("random36", "shuffled36"))

integrity = len(complete) >= 10 and not invalid
mean_recovery = mean36 is not None and mean36 >= 58 and recovered36 >= 12
teacher_specific = specific is not None and specific["bootstrap95"][0] > 4
verdict = ("REPLICATED" if integrity and mean_recovery and teacher_specific else
           "PARTIAL" if integrity and (mean_recovery != teacher_specific) else "NOT REPLICATED")

summary = {
    "protocol": "docs/advice-budget-replication-protocol.md",
    "finalSeeds": list(SEEDS),
    "teacherGatePassed": len(complete),
    "teacherReadinessFailures": [{"seed": s, "teacherWins": r["teacherWins"]} for s, r in zip(SEEDS, results) if r["status"] != "completed"],
    "invalidSeeds": invalid,
    "messagesDecodedPerCompletedSeed": EXPECTED if complete and not invalid else None,
    "meanWins": {"48": {a: statistics.mean(row[a] for row in rows48) for a in arms} if rows48 else None,
                 "24": {a: statistics.mean(row[a] for row in rows24) for a in arms} if rows24 else None},
    "recoveredAt48": {a: sum(row[a] >= 58 for row in rows48) for a in arms},
    "paired48": {
        "trained36MinusStrongerMatchedControl": specific,
        "trained36MinusRandom36": paired(rows48, "trained36", "random36"),
        "trained36MinusAlone": paired(rows48, "trained36", "alone"),
        "trained46MinusAlone": paired(rows48, "trained46", "alone"),
    },
    "criteria": {"integrity": integrity, "meanRecovery": mean_recovery, "teacherSpecificBenefit": teacher_specific},
    "verdict": verdict,
    "seedResults": [{"seed": s, "status": r["status"], "teacherWins": r["teacherWins"],
                     "messages": r.get("messages"), "scores24": at(r, 24) if r["status"] == "completed" else None,
                     "scores48": at(r, 48) if r["status"] == "completed" else None,
                     "runtimeSeconds": r.get("runtimeSeconds")} for s, r in zip(SEEDS, results)],
}
output = ROOT / "data/advice_budget_replication_summary.json"
with output.open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({k: summary[k] for k in ("teacherGatePassed", "invalidSeeds", "meanWins", "recoveredAt48", "paired48", "criteria", "verdict")}, indent=1))
