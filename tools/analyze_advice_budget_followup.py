"""Locked analysis for docs/advice-budget-followup-protocol.md; final seeds 100-114 only."""
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (12, 24, 36, 46)
SEEDS = range(100, 115)
results = [json.loads((ROOT / f"data/advice_budget_followup/seed_{s}.json").read_text()) for s in SEEDS]
complete = [r for r in results if r["status"] == "completed"]
invalid = []
for s, r in zip(SEEDS, results):
    if r["status"] != "completed":
        continue
    expected = {f"{kind}{b}": b for b in BUDGETS for kind in ("trained", "random", "shuffled")}
    if (r["messages"] != expected or not r["teacherFrozen"] or not r["synapsesUnchanged"] or
            [p["rounds"] for p in r["points"]] != [0, 24, 48]):
        invalid.append(s)


def interval(differences):
    rng = random.Random(2026111515)
    means = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10_000)]
    means.sort()
    return [means[250], means[9750]]


def final(r):
    return r["points"][-1]


summary = {
    "protocol": "docs/advice-budget-followup-protocol.md",
    "finalSeeds": list(SEEDS),
    "teacherGatePassed": len(complete),
    "teacherGateFailed": len(results) - len(complete),
    "invalidSeeds": invalid,
    "baselineMean": statistics.mean(final(r)["alone"] for r in complete) if complete else None,
    "budgets": {},
    "seedResults": [],
}
for b in BUDGETS:
    name, random_name, shuffled_name = (f"{kind}{b}" for kind in ("trained", "random", "shuffled"))
    rows = [final(r) for r in complete]
    scores = [row[name] for row in rows]
    vs_alone = [row[name] - row["alone"] for row in rows]
    vs_control = [row[name] - max(row[random_name], row[shuffled_name]) for row in rows]
    ci_alone, ci_control = interval(vs_alone), interval(vs_control)
    summary["budgets"][str(b)] = {
        "messagesPerCompletedSeed": b,
        "means": {kind: statistics.mean(row[f"{kind}{b}"] for row in rows)
                  for kind in ("trained", "random", "shuffled")},
        "recovered": sum(x >= 58 for x in scores),
        "pairedVsAlone": {"mean": statistics.mean(vs_alone), "bootstrap95": ci_alone},
        "pairedVsStrongerMatchedControl": {
            "mean": statistics.mean(vs_control), "bootstrap95": ci_control},
        "reliableInSample": len(complete) == 15 and not invalid and
            sum(x >= 58 for x in scores) >= 12 and statistics.mean(scores) >= 58 and
            ci_control[0] > 4,
    }
for s, r in zip(SEEDS, results):
    summary["seedResults"].append({"seed": s, "status": r["status"],
        "teacherWins": r["teacherWins"],
        "messages": r.get("messages"),
        "scores48": final(r) if r["status"] == "completed" else None})

output = ROOT / "data/advice_budget_followup_summary.json"
with output.open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({"gatePassed": summary["teacherGatePassed"],
                  "invalid": invalid, "budgets": summary["budgets"]}))
