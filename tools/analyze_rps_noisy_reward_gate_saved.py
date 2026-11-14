"""Read-only analysis of the archived noisy reward-gating prerequisite run.

Run once: python tools/analyze_rps_noisy_reward_gate_saved.py
This never invokes the simulator or changes the source result.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_path = root / "data/rps_noisy_reward_gate_prerequisites.json"
output_path = root / "data/rps_noisy_reward_gate_early_headroom.json"
source = json.loads(source_path.read_text(encoding="utf-8"))
if source["status"] != "stopped_receiver" or len(source["students"]) != 2:
    raise ValueError("Archived prerequisite result does not match the expected run")


def margin(row):
    cue = row["cue"]
    return row["scores"][cue] - max(
        score for action, score in enumerate(row["scores"]) if action != cue
    )


def summarize_window(rows):
    return {
        "observed": len(rows),
        "rewardedOptimal": sum(r["action"] == r["cue"] and r["reward"] == 1 for r in rows),
        "misleadingPositive": sum(r["action"] != r["cue"] and r["reward"] == 1 for r in rows),
        "positiveOnlineQMargin": sum(margin(r) > 0 for r in rows),
    }


student_results = []
for student in source["students"]:
    rows = [
        {**row, "globalRound": episode["episode"] * 24 + row["round"] + 1}
        for episode in student["training"]
        for row in episode["rows"]
        if row["cue"] is not None
    ]
    assert len(rows) == 276 and all(r["estimatedCue"] == r["cue"] for r in rows)
    by_cue = []
    for cue in range(3):
        cue_rows = [r for r in rows if r["cue"] == cue]
        optimal_rewards = [
            r for r in cue_rows if r["action"] == cue and r["reward"] == 1
        ]
        misleading = [
            r for r in cue_rows if r["action"] != cue and r["reward"] == 1
        ]
        first_optimal = optimal_rewards[0]["globalRound"] if optimal_rewards else None
        positive_steps = [
            r for r in misleading if .03 * (1 - r["scores"][r["action"]]) > 1e-12
        ]
        by_cue.append({
            "cue": cue,
            "trainingExposures": len(cue_rows),
            "firstRewardedOptimalRound": first_optimal,
            "firstMisleadingPositiveRound": misleading[0]["globalRound"] if misleading else None,
            "rewardedOptimal": len(optimal_rewards),
            "misleadingPositive": len(misleading),
            "misleadingPositiveWeightIncreasingSteps": len(positive_steps),
            "misleadingBeforeFirstOptimalReward": sum(
                r["globalRound"] < first_optimal for r in misleading
            ) if first_optimal is not None else None,
            "misleadingAtNonpositiveOnlineMargin": sum(margin(r) <= 0 for r in misleading),
            "misleadingAtPositiveOnlineMargin": sum(margin(r) > 0 for r in misleading),
            "positiveOnlineMarginByEpisode": [
                {"episode": episode, "positive": sum(
                    margin(r) > 0 for r in cue_rows
                    if (r["globalRound"] - 1) // 24 == episode
                ), "n": sum((r["globalRound"] - 1) // 24 == episode for r in cue_rows)}
                for episode in range(12)
            ],
        })
    student_results.append({
        "seed": student["seed"],
        "initialFrozenOptimal": student["initial"]["summary"]["expectedOptimal"],
        "finalFrozenOptimal": student["final"]["summary"]["expectedOptimal"],
        "frozenObserved": 69,
        "intermediateFrozenCheckpoints": 0,
        "intermediateFullControllerSnapshots": 0,
        "perTrialPreActionQScoresAvailable": True,
        "trainingSensoryRouting": {"correct": len(rows), "n": len(rows)},
        "windows": {str(end): summarize_window(
            [r for r in rows if r["globalRound"] <= end]
        ) for end in (24, 72, 288)},
        "byCue": by_cue,
    })

source_results = []
for teacher in source["source"]:
    rows = [r for r in teacher["rows"] if r["candidate"] is not None]
    by_cue = []
    for cue in range(3):
        cue_rows = [r for r in rows if r["cue"] == cue]
        by_action = []
        for action in range(3):
            x = [r for r in cue_rows if r["candidate"] == action]
            by_action.append({
                "action": action,
                "n": len(x),
                "approvals": sum(r["bit"] == 1 for r in x),
                "rewarded": sum(r["candidateReward"] == 1 for r in x),
            })
        by_cue.append({"cue": cue, "byCandidateAction": by_action})
    windows = {}
    for end in (24, 72, 288):
        x = [r for r in rows if r["round"] < end]
        lucky = [r for r in x if not r["expectedOptimalCandidate"]
                 and r["candidateReward"] == 1]
        good = [r for r in x if r["expectedOptimalCandidate"]
                and r["candidateReward"] == 1]
        windows[str(end)] = {
            "observedCandidates": len(x),
            "misleadingPositive": len(lucky),
            "rejectedMisleading": sum(r["bit"] == 0 for r in lucky),
            "rewardedOptimal": len(good),
            "approvedRewardedOptimal": sum(r["bit"] == 1 for r in good),
        }
    source_results.append({
        "teacherSeed": teacher["seed"],
        "matchedToStudentTrainingTrialIDs": False,
        "teacherState": "frozen controller, independent continuous 288-round source stream",
        "windows": windows,
        "byCue": by_cue,
    })

receiver_results = []
for seed, cue, phase in ((863, 1, "second"), (864, 0, "first")):
    group = [p for p in source["receiver"]["probes"]
             if p["seed"] == seed and p["cue"] == cue]
    branches = []
    for bit in (None, 0, 1):
        x = [p for p in group if p["bit"] == bit]
        branches.append({
            "bit": bit,
            "recognized": sum(p[phase]["recognized"] for p in x),
            "vpnHeadCorrect": sum(p[phase]["estimatedCue"] == p[phase]["cue"] for p in x),
            "errorRepeats": [p["repeat"] for p in x if not p[phase]["recognized"]],
        })
    receiver_results.append({
        "seed": seed, "firstCue": cue, "phase": phase,
        "branches": branches,
        "sameErrorsAcrossBranches": all(
            b["errorRepeats"] == branches[0]["errorRepeats"] for b in branches[1:]
        ),
        "resetBeforeSecondCue": True,
        "timingAlternativeTested": False,
    })

result = {
    "kind": "read_only_saved_result_diagnostic",
    "source": source_path.name,
    "method": {
        "earlyWindows": "training rounds 1-24 and 1-72; final 1-288, fixed before calculation",
        "onlineMargin": "pre-action Q_expectedOptimal minus largest competing pre-action Q",
        "misleadingPositive": "reward +1 for an action suboptimal in expectation under noise 0.5",
        "weightIncreasingStep": "0.03*(1-Q_played)>1e-12 on a misleading positive-reward trial",
        "caveat": "Training was balanced and actions were forced; online Q margins are not frozen checkpoint accuracy.",
    },
    "students": student_results,
    "teacherSource": source_results,
    "receiverFailures": receiver_results,
    "limits": [
        "Only initial and final frozen student evaluations were saved.",
        "No teacher Q/neural observations were aligned to the students' actual training trials; source-trace selectivity cannot be assigned as observed advice for those trials.",
        "The receiver's first cue is classified before DN stimulation; its second cue follows a full student neural-state reset.",
        "No reward-gated student policy was simulated or trained.",
    ],
}
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps({
    "students": [{"seed": x["seed"], "windows": x["windows"],
                  "byCue": [{k: c[k] for k in ("cue", "firstRewardedOptimalRound",
                      "firstMisleadingPositiveRound", "misleadingPositive",
                      "misleadingBeforeFirstOptimalReward")}
                      for c in x["byCue"]]} for x in student_results],
    "teacherSource": [{"seed": x["teacherSeed"], "windows": x["windows"]}
                      for x in source_results],
    "receiverFailures": receiver_results,
}, indent=2))
