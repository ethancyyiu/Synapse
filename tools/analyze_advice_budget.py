"""Analyze only the locked final seeds of docs/advice-budget-protocol.md.

Writes one new summary; never reads the pilot or replaces an existing result.
"""

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "advice_budget"
RUNS = [json.loads((DATA / f"seed_{s}.json").read_text(encoding="utf-8")) for s in range(100, 115)]
VALID_MESSAGES = {"full": 46, "early12": 12, "spaced12": 12, "random12": 12}
completed = [r for r in RUNS if r["status"] == "completed"]
invalid = [
    r["studentSeed"]
    for r in completed
    if r["messages"] != VALID_MESSAGES
    or not r["teacherFrozen"]
    or not r["synapsesUnchanged"]
    or [p["rounds"] for p in r["points"]] != [0, 24, 48]
]


def point(run, rounds):
    return next(p for p in run["points"] if p["rounds"] == rounds)


contrasts = [point(r, 48)["spaced12"] - max(point(r, 48)["alone"], point(r, 48)["random12"]) for r in completed]
if contrasts:
    gen = random.Random(2026111512)
    samples = sorted(statistics.mean(gen.choices(contrasts, k=len(contrasts))) for _ in range(10_000))
    interval = [samples[249], samples[9749]]
    mean_contrast = statistics.mean(contrasts)
else:
    interval = [None, None]
    mean_contrast = None

arms = ["alone", "full", "early12", "spaced12", "random12"]
means = {
    str(rounds): {arm: statistics.mean(point(r, rounds)[arm] for r in completed) for arm in arms}
    for rounds in (24, 48)
} if completed else {}
mean_spaced = means["48"]["spaced12"] if completed else None
passed = (
    len(completed) >= 10
    and not invalid
    and mean_spaced >= 58
    and interval[0] > 4
)
summary = {
    "protocol": "docs/advice-budget-protocol.md",
    "seedsRun": 15,
    "teacherGatePassed": len(completed),
    "teacherGateFailed": 15 - len(completed),
    "invalidStudentSeeds": invalid,
    "meanWins": means,
    "primary": {
        "contrast": "spaced12 - max(alone, random12) at 48 rounds",
        "mean": mean_contrast,
        "bootstrap95": interval,
        "passed": passed,
    },
    "secondaryDescriptive": {
        "early12MinusSpaced12": means["48"]["early12"] - mean_spaced if completed else None,
        "fullMinusSpaced12": means["48"]["full"] - mean_spaced if completed else None,
    },
    "seedResults": [
        {"studentSeed": r["studentSeed"], "teacherWins": r["teacherWins"], "status": r["status"],
         "at48": point(r, 48) if r["status"] == "completed" else None}
        for r in RUNS
    ],
}
out = ROOT / "data" / "advice_budget_summary.json"
with out.open("x", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
    file.write("\n")
print(json.dumps({"teacherGatePassed": len(completed), "meanWins48": means.get("48"),
                  "primary": summary["primary"], "summary": str(out.relative_to(ROOT))}))
