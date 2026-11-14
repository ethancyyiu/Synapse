"""Summarize locked-source teacher activity and independent Fly B tests.

Run after all recordings: python tools/summarize_rps_selected_teacher.py
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_rps_teacher_activity import conditional_information

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (61, 62, 63)
KINDS = ("alone", "trained", "untrained", "shuffled")
COLORS = {"alone": "#555555", "trained": "#1769aa", "untrained": "#c77716",
          "shuffled": "#909090"}


def mean(values):
    return float(np.mean(values))


def recovery_rounds(events):
    wins = np.array([event["reward"] == 1 for event in events], dtype=float)[240:]
    # A 72-round window must exceed 50% and stay there for the next 72 endpoints.
    for end in range(72, len(wins) - 71):
        if all(wins[later - 72:later].mean() >= .5 for later in range(end, end + 72)):
            return end
    return None


def cell_summary(runs, kind):
    early24 = [run["conditions"][kind]["earlyPostSwitch24"] for run in runs]
    early72 = [run["conditions"][kind]["earlyPostSwitch72"] for run in runs]
    frozen = [mean(run["conditions"][kind]["frozenDisconnected"]) for run in runs]
    post = [mean([event["reward"] == 1 for event in run["conditions"][kind]["events"][240:]])
            for run in runs]
    recovery = [recovery_rounds(run["conditions"][kind]["events"]) for run in runs]
    return {"early24BySeed": early24, "early24Mean": mean(early24),
            "early72BySeed": early72, "early72Mean": mean(early72),
            "postSwitchBySeed": post, "postSwitchMean": mean(post),
            "frozenDisconnectedBySeed": frozen, "frozenDisconnectedMean": mean(frozen),
            "stableRecoveryRoundsBySeed": recovery}


def signal_effect(runs, kind):
    rows = [event for run in runs for event in run["conditions"][kind]["events"]
            if event["signal"] is not None]
    label = np.array([event["signal"] for event in rows], dtype=np.int8)
    action = np.array([event["action"] for event in rows], dtype=np.int8)
    prev = np.array([3 if event["observation"] is None else event["observation"]
                     for event in rows], dtype=np.int8)
    target = np.array([event["targetSpikes"] for event in rows], dtype=float)
    rates = np.array([event["rates"] for event in rows], dtype=float)
    return {"rounds": len(rows), "labelZeroFraction": mean(label == 0),
            "targetSpikesMeanByLabel": {str(k): target[label == k].mean(axis=0).tolist()
                                         for k in (0, 1)},
            "mbonRatesMeanByLabel": {str(k): rates[label == k].mean(axis=0).tolist()
                                      for k in (0, 1)},
            "actionProbabilitiesByLabel": {str(k): (np.bincount(action[label == k], minlength=3)
                                           / max(1, int((label == k).sum()))).tolist() for k in (0, 1)},
            "actionSignalInformationGivenPreviousMoveBits": float(
                conditional_information(label[:, None], action, prev)[0])}


def main():
    runs = {mode: [json.loads((ROOT / "data" / f"rps_selected_teacher_{mode}_{seed}.json")
                            .read_text()) for seed in SEEDS] for mode in ("full", "postonly")}
    for mode, cells in runs.items():
        if [run["seed"] for run in cells] != list(SEEDS):
            raise RuntimeError("Evaluation seed mismatch")
    activity = json.loads((ROOT / "data" / "rps_teacher_activity_evaluation_analysis.json").read_text())
    calibration = json.loads((ROOT / "data" / "rps_teacher_activity_calibration_analysis.json").read_text())
    random_wiring = json.loads((ROOT / "data" / "rps_teacher_activity_calibration_shuffled_analysis.json")
                               .read_text())
    result = {"seeds": SEEDS, "calibrationSeeds": (41, 42, 43),
              "lockedSource": activity["lockedSource"],
              "sourcePermutation": activity["lockedSourcePermutation"],
              "sourceEpisodeFrequency": activity["lockedSourceEpisodeFrequency"],
              "randomWiringOneSeed": random_wiring["lockedSource"],
              "calibrationAssociationLeaders": calibration["associationLeaders"],
              "students": {}}
    recordings = [json.load(gzip.open(ROOT / "data" / f"rps_teacher_activity_evaluation_{seed}.json.gz",
                                       "rt")) for seed in SEEDS]
    no_switch = [json.loads((ROOT / "data" / f"rps_teacher_noswitch_{seed}.json").read_text())
                 for seed in SEEDS]
    for record, control in zip(recordings, no_switch):
        local = record["neuronIds"].index(2132)
        recorded_bits = [int(next((value for j, value in row["traces"] if j == local), 0) >= .5)
                         for row in record["trained"]["samples"][:240]]
        if recorded_bits != control["sourceBits"][:240]:
            raise RuntimeError("No-switch control did not match the trained teacher before switch")
    result["noSwitchControl"] = {
        "episodeFrequencyMean": np.mean([run["sourceFrequency"] for run in no_switch], axis=0).tolist(),
        "lateSourceFrequencyBySeed": [mean(run["sourceFrequency"][16:20]) for run in no_switch],
        "lateWinRateBySeed": [mean(run["episodeRates"][16:20]) for run in no_switch],
    }
    result["teacherPerformance"] = {
        "pretrainingLast3": [mean(record["pretrainingRates"][-3:]) for record in recordings],
        **{condition: {
            "preSwitchLast3": [mean(record[condition]["episodeRates"][7:10]) for record in recordings],
            "postSwitchFirst3": [mean(record[condition]["episodeRates"][10:13]) for record in recordings],
            "postSwitchLast3": [mean(record[condition]["episodeRates"][17:20]) for record in recordings],
        } for condition in ("trained", "untrained")},
    }
    for mode, cells in runs.items():
        result["students"][mode] = {kind: cell_summary(cells, kind) for kind in KINDS}
        result["students"][mode]["signalEffect"] = {
            kind: signal_effect(cells, kind) for kind in KINDS[1:]}
        for metric in ("early24", "early72", "frozenDisconnected"):
            result["students"][mode][f"trainedMinusControls_{metric}"] = {
                kind: [round(a - b, 6) for a, b in zip(
                    result["students"][mode]["trained"][f"{metric}BySeed"],
                    result["students"][mode][kind][f"{metric}BySeed"])]
                for kind in ("alone", "untrained", "shuffled")}

    # The selected bit versus the previous result Fly B itself received without a teacher.
    labels, next_moves, contexts = [], [], []
    for run in runs["full"]:
        events = run["conditions"]["alone"]["events"]
        for i, event in enumerate(events):
            previous_reward = 3 if event["round"] == 0 else events[i - 1]["reward"] + 1
            previous_move = 3 if event["observation"] is None else event["observation"]
            labels.append(int(run["trainedLabels"][i] == 0))
            next_moves.append(event["opponent"])
            contexts.append(previous_move * 4 + previous_reward)
    bit = np.array(labels)[:, None]
    next_moves = np.array(next_moves)
    contexts = np.array(contexts)
    observed_extra = float(conditional_information(bit, next_moves, contexts)[0])
    random = np.random.default_rng(20261102)
    null_extra = []
    for _ in range(500):
        permuted = bit.copy()
        for context in np.unique(contexts):
            group = np.flatnonzero(contexts == context)
            permuted[group, 0] = random.permutation(permuted[group, 0])
        null_extra.append(float(conditional_information(permuted, next_moves, contexts)[0]))
    result["lockedSource"]["nextGivenStudentPreviousMoveAndPreviousRewardBits"] = observed_extra
    result["lockedSource"]["studentObservationConditionedPermutation"] = {
        "null95Bits": float(np.quantile(null_extra, .95)),
        "permutationP": (1 + sum(value >= observed_extra for value in null_extra)) / 501,
    }

    output = ROOT / "data" / "rps_selected_teacher_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    freq = activity["lockedSourceEpisodeFrequency"]
    x = np.arange(1, 21)
    for condition, color in (("trained", COLORS["trained"]), ("untrained", COLORS["untrained"])):
        axes[0].plot(x, freq[condition], marker="o", ms=3, lw=2, color=color, label=condition)
    axes[0].plot(x, result["noSwitchControl"]["episodeFrequencyMean"], lw=2,
                 color=COLORS["trained"], ls="--", label="trained, no switch")
    axes[0].axvline(10.5, color="black", ls="--", lw=1)
    axes[0].set(xlabel="Episode", ylabel="Fraction of rounds with source bit = 0",
                title="Locked Fly A neuron 2132, 200 ms trace at 120 ms")
    axes[0].legend()
    for kind in KINDS:
        wins = np.array([[event["reward"] == 1 for event in run["conditions"][kind]["events"][240:]]
                         for run in runs["full"]], dtype=float)
        curve = np.convolve(wins.mean(axis=0), np.ones(24) / 24, mode="valid")
        axes[1].plot(np.arange(24, 241), curve, lw=1.8, color=COLORS[kind], label=kind)
    axes[1].axhline(1 / 3, color="black", ls=":", lw=1, label="chance")
    axes[1].set(xlabel="Rounds after strategy switch (24-round rolling window)",
                ylabel="Win rate", title="Fly B on independent evaluation seeds")
    axes[1].legend(ncol=3, fontsize=8)
    figure = ROOT / "docs" / "rps-diagnostics" / "selected-teacher-source.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    print(f"Saved {output} and {figure}")
    for mode in ("full", "postonly"):
        print(mode)
        for kind in KINDS:
            cell = result["students"][mode][kind]
            print(kind, "early24", round(cell["early24Mean"], 3), "early72",
                  round(cell["early72Mean"], 3), "frozen",
                  round(cell["frozenDisconnectedMean"], 3),
                  "recovery", cell["stableRecoveryRoundsBySeed"])


if __name__ == "__main__":
    main()
