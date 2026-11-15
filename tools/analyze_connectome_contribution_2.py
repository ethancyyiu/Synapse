"""Locked analysis for docs/connectome-contribution-2-protocol.md (Study 1).

python tools/analyze_connectome_contribution_2.py diagnostic   -> data/endogenous_signal_summary.json
python tools/analyze_connectome_contribution_2.py teaching     -> data/connectome_contribution_2_summary.json
"""
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_once(path, summary):
    with (ROOT / path).open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def centroid_decoder(train_x, train_y):
    dims = len(train_x[0])
    mean = [statistics.mean(x[j] for x in train_x) for j in range(dims)]
    std = [statistics.pstdev([x[j] for x in train_x]) or 1.0 for j in range(dims)]
    z = lambda x: [(x[j] - mean[j]) / std[j] for j in range(dims)]
    cents = {}
    for label in set(train_y):
        pts = [z(x) for x, y in zip(train_x, train_y) if y == label]
        cents[label] = [statistics.mean(p[j] for p in pts) for j in range(dims)]
    return lambda x: min(cents, key=lambda c: sum((a - b) ** 2 for a, b in zip(z(x), cents[c])))


def diagnostic():
    seeds = []
    for s in range(5):
        d = json.loads((ROOT / f"data/endogenous_signal_v2/seed_{s}.json").read_text())
        rows = d["rows"]
        cal = [r for r in rows if r["episode"] < 3 and r["cue"] is not None]
        test = [r for r in rows if r["episode"] >= 3 and r["cue"] is not None]
        feat = lambda m: [math.log1p(v) for v in m]
        # (i) cue only
        majority = {}
        for cue in {r["cue"] for r in cal}:
            labels = [r["advice"] for r in cal if r["cue"] == cue]
            majority[cue] = max(set(labels), key=labels.count)
        acc = lambda preds: sum(p == r["advice"] for p, r in zip(preds, test)) / len(test)
        cue_acc = acc([majority.get(r["cue"]) for r in test])
        # (ii) trained activity, (iii) untrained activity with the same decoder
        dec = centroid_decoder([feat(r["trainedMbon"]) for r in cal], [r["advice"] for r in cal])
        neural_acc = acc([dec(feat(r["trainedMbon"])) for r in test])
        untrained_acc = acc([dec(feat(r["untrainedMbon"])) for r in test])
        gdec = centroid_decoder([r["trainedGroupRates"] for r in cal], [r["advice"] for r in cal])
        group_acc = acc([gdec(r["trainedGroupRates"]) for r in test])
        group_untrained_acc = acc([gdec(r["untrainedGroupRates"]) for r in test])
        # (iv) permutation null
        rng = random.Random(2026111551 + s)
        null = []
        for _ in range(200):
            labels = [r["advice"] for r in cal]
            rng.shuffle(labels)
            if len(set(labels)) < 1:
                continue
            pdec = centroid_decoder([feat(r["trainedMbon"]) for r in cal], labels)
            null.append(acc([pdec(feat(r["trainedMbon"])) for r in test]))
        null.sort()
        seeds.append({"seed": s, "gatePassed": d["gatePassed"], "teacherGateWins": d["teacherGateWins"],
                      "heldOutRounds": len(test),
                      "identicalAllNeuronsShare": sum(r["identicalAllNeurons"] for r in rows) / len(rows),
                      "identicalMbonShare": sum(r["trainedMbon"] == r["untrainedMbon"] for r in rows) / len(rows),
                      "adviceDiffersFromUntrainedShare": sum(r["advice"] != r["untrainedAction"] for r in rows) / len(rows),
                      "accuracy": {"cueOnly": cue_acc, "trainedActivity": neural_acc, "untrainedActivity": untrained_acc,
                                   "trainedGroupRates": group_acc, "untrainedGroupRates": group_untrained_acc,
                                   "permutationNull95th": null[int(0.95 * len(null)) - 1] if null else None}})
    gated = [x for x in seeds if x["gatePassed"]]
    gain_cue = statistics.mean(x["accuracy"]["trainedActivity"] - x["accuracy"]["cueOnly"] for x in gated) if gated else None
    gain_learned = statistics.mean(x["accuracy"]["trainedActivity"] - x["accuracy"]["untrainedActivity"] for x in gated) if gated else None
    proceed = bool(gated) and gain_cue >= 0.10 and gain_learned >= 0.10
    summary = {"protocol": "docs/connectome-contribution-2-protocol.md", "part": "E diagnostic", "seeds": seeds,
               "gatedSeeds": len(gated), "meanGainOverCueOnly": gain_cue, "meanGainOverUntrainedActivity": gain_learned,
               "decision": "PROCEED to teaching with arm E" if proceed else "STOP arm E: no endogenous learned teaching signal found"}
    write_once("data/endogenous_signal_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("gatedSeeds", "meanGainOverCueOnly", "meanGainOverUntrainedActivity", "decision")}, indent=1))
    for x in seeds:
        print(x["seed"], x["gatePassed"], round(x["identicalAllNeuronsShare"], 3), {k: (round(v, 3) if v is not None else None) for k, v in x["accuracy"].items()})


def teaching():
    SEEDS = range(15)
    BOOT = 2026111550
    results = [json.loads((ROOT / f"data/connectome_contribution_2/seed_{s}.json").read_text()) for s in SEEDS]

    def at(world, rounds):
        return next(p for p in world["points"] if p["rounds"] == rounds)

    def ci(diffs):
        rng = random.Random(BOOT)
        means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10_000))
        return {"mean": statistics.mean(diffs), "bootstrap95": [means[250], means[9750]], "n": len(diffs), "perSeed": diffs}

    real = [r["worlds"]["real"] for r in results if r["worlds"]["real"]["status"] == "completed"]
    both = [r["worlds"] for r in results if r["worlds"]["real"]["status"] == "completed" and r["worlds"]["rewired"]["status"] == "completed"]
    comm = lambda w, arm: w["counts"][arm]["decoded"] / max(1, w["counts"][arm]["sent"])
    q1 = ci([at(w, 48)["neural"] - at(w, 48)["digital"] for w in real]) if real else None
    q2 = ci([at(w, 48)["neural"] - at(w, 48)["random"] for w in real]) if real else None
    q3 = ci([(at(w["real"], 48)["neural"] - at(w["real"], 48)["alone"]) - (at(w["rewired"], 48)["neuralS"] - at(w["rewired"], 48)["aloneS"]) for w in both]) if both else None
    comm_ok = all(comm(w["real"], "neural") >= 0.99 and comm(w["rewired"], "neuralS") >= 0.99 for w in both)
    summary = {
        "protocol": "docs/connectome-contribution-2-protocol.md", "part": "A-D teaching",
        "realCompleted": len(real), "rewiredCompleted": len(both),
        "teacherGates": [{"seed": s, "real": r["worlds"]["real"].get("teacherWins"), "rewired": r["worlds"]["rewired"].get("teacherWins")} for s, r in zip(SEEDS, results)],
        "communication": {"realNeural": [comm(w, "neural") for w in real],
                          "rewiredNeural": [comm(w["rewired"], "neuralS") for w in both]},
        "meanWins48": {"real": {a: statistics.mean(at(w, 48)[a] for w in real) for a in ("alone", "neural", "digital", "random")} if real else None,
                       "rewired": {a: statistics.mean(at(w["rewired"], 48)[a] for w in both) for a in ("aloneS", "neuralS", "digitalS")} if both else None},
        "q1_neural_minus_digital_48": q1,
        "q1_verdict": None if not q1 else ("spikes add something" if q1["bootstrap95"][0] > 4 else
                                           "equivalent within +/-4" if q1["bootstrap95"][0] >= -4 and q1["bootstrap95"][1] <= 4 else "inconclusive"),
        "q2_neural_minus_random_48": q2,
        "q3_realWiringTeachingContribution_48": q3,
        "q3_verdict": ("stopped at teacher readiness" if len(both) < 10 else
                       "real wiring contributes to teaching" if q3["bootstrap95"][0] > 4 and comm_ok else "no contribution shown"),
        "reported": {"alone_real_minus_aloneS": ci([at(w["real"], 48)["alone"] - at(w["rewired"], 48)["aloneS"] for w in both]) if both else None,
                     "digitalS_minus_neuralS": ci([at(w["rewired"], 48)["digitalS"] - at(w["rewired"], 48)["neuralS"] for w in both]) if both else None},
    }
    write_once("data/connectome_contribution_2_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("realCompleted", "rewiredCompleted", "meanWins48", "q1_verdict", "q3_verdict")}, indent=1))
    for k in ("q1_neural_minus_digital_48", "q2_neural_minus_random_48", "q3_realWiringTeachingContribution_48"):
        v = summary[k]
        print(k, None if not v else (round(v["mean"], 2), [round(x, 2) for x in v["bootstrap95"]]))


if __name__ == "__main__":
    {"diagnostic": diagnostic, "teaching": teaching}[sys.argv[1]]()
