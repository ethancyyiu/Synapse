"""Analyze locked-source pretrained teaching runs; no source or channel tuning.

Run: python tools/analyze_rps_pretrained_teacher.py 101 102 103 104 105 106
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_rps_teacher_activity import conditional_information

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("alone", "pretrained", "simultaneous", "untrained", "shuffled")
COLORS = {"alone": "#555555", "pretrained": "#0969a5", "simultaneous": "#c46c1a",
          "untrained": "#7b4ba7", "shuffled": "#999999"}
RANDOM = np.random.default_rng(20261104)


def mean(values):
    return float(np.mean(values))


def bootstrap_ci(values, repeats=10000):
    values = np.asarray(values, dtype=float)
    draws = RANDOM.choice(values, (repeats, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def entropy(bit):
    p = mean(bit == 0)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p)) if 0 < p < 1 else 0.0


def information_test(bit, target, contexts, repeats=500):
    bit = np.asarray(bit, dtype=np.uint8)
    target = np.asarray(target, dtype=np.int8)
    contexts = np.asarray(contexts, dtype=np.int16)
    observed = float(conditional_information(bit[:, None], target, contexts)[0])
    null = []
    for _ in range(repeats):
        permuted = bit.copy()
        for context in np.unique(contexts):
            indices = np.flatnonzero(contexts == context)
            permuted[indices] = RANDOM.permutation(permuted[indices])
        null.append(float(conditional_information(permuted[:, None], target, contexts)[0]))
    return {"bits": observed, "permutationNull95Bits": float(np.quantile(null, .95)),
            "permutationP": (1 + sum(value >= observed for value in null)) / (repeats + 1)}


def teacher_info(runs, kind):
    rows = [row for run in runs for row in run["teacherStreams"][kind]["rows"]]
    bit = np.array([row["bit"] for row in rows], dtype=np.uint8)
    correct = np.array([row["correctAction"] for row in rows], dtype=np.int8)
    teacher_action = np.array([row["teacherAction"] for row in rows], dtype=np.int8)
    previous = np.array([3 if row["previousOpponent"] is None else row["previousOpponent"]
                         for row in rows], dtype=np.int8)
    episode = np.array([row["episode"] - 10 for row in rows], dtype=np.int8)
    seed_group = np.repeat(np.arange(len(runs), dtype=np.int16), 240)
    b_reward = np.array([3 if event["previousReward"] is None else event["previousReward"] + 1
                         for run in runs for event in run["conditions"]["alone"]["postEvents"]],
                        dtype=np.int8)
    if len(b_reward) != len(bit):
        raise RuntimeError("Teacher and student rounds do not align")
    by_previous = {str(p): {"zero": int(((previous == p) & (bit == 0)).sum()),
                            "one": int(((previous == p) & (bit == 1)).sum())}
                   for p in range(4)}
    entropy_given_previous = sum(mean(previous == p) * entropy(bit[previous == p])
                                 for p in range(4))
    return {"rounds": len(rows), "bitZeroFraction": mean(bit == 0), "bitEntropyBits": entropy(bit),
            "bitCountsByPreviousMove": by_previous,
            "bitEntropyGivenPreviousMoveBits": entropy_given_previous,
            "bitZeroBySeed": [mean(bit[i * 240:(i + 1) * 240] == 0)
                              for i in range(len(runs))],
            "correctActionGivenPreviousMoveBitsBySeed": [float(conditional_information(
                bit[i * 240:(i + 1) * 240, None], correct[i * 240:(i + 1) * 240],
                previous[i * 240:(i + 1) * 240])[0]) for i in range(len(runs))],
            "bitZeroByEpisode": [mean(bit[episode == ep] == 0) for ep in range(10)],
            "bitZeroByCorrectAction": [mean(bit[correct == action] == 0)
                                       for action in range(3)],
            "correctActionInformation": information_test(bit, correct,
                                                              np.zeros(len(bit), dtype=np.int8)),
            "correctActionGivenPreviousMove": information_test(bit, correct, previous),
            "correctActionGivenPreviousMoveAndSeed": information_test(
                bit, correct, seed_group * 4 + previous),
            "teacherActionGivenPreviousMove": information_test(bit, teacher_action, previous),
            "correctActionGivenPreviousMoveAndBReward": information_test(
                bit, correct, previous * 4 + b_reward),
            "correctActionGivenPreviousMoveBRewardAndSeed": information_test(
                bit, correct, seed_group * 16 + previous * 4 + b_reward),
            "teacherWinRatesBySeed": [mean(run["teacherStreams"][kind]["episodeWinRates"])
                                      for run in runs],
            "teacherWeightsChanged": [run["teacherStreams"][kind]["weightsChanged"]
                                      for run in runs]}


def recovery_rounds(events):
    wins = np.array([event["reward"] == 1 for event in events], dtype=float)
    for end in range(72, len(wins) - 71):
        if all(wins[later - 72:later].mean() >= .5 for later in range(end, end + 72)):
            return end
    return None


def student_metrics(runs, kind):
    per_seed = []
    for run in runs:
        cell = run["conditions"][kind]
        wins = np.array([event["reward"] == 1 for event in cell["postEvents"]], dtype=float)
        frozen = cell["frozenDisconnected"]
        per_seed.append({"early24": mean(wins[:24]), "early72": mean(wins[:72]),
                         "post240": mean(wins), "frozenDisconnected": mean(frozen),
                         "recoveryRounds": recovery_rounds(cell["postEvents"])})
    return {"bySeed": per_seed,
            **{metric: {"mean": mean([row[metric] for row in per_seed]),
                        "bootstrap95": bootstrap_ci([row[metric] for row in per_seed])}
               for metric in ("early24", "early72", "post240", "frozenDisconnected")}}


def signal_activity(runs, kind):
    rows = [(run["seed"], event) for run in runs
            for event in run["conditions"][kind]["postEvents"]]
    bits = np.array([event["bit"] for _, event in rows], dtype=np.uint8)
    prev = np.array([3 if event["previousOpponent"] is None else event["previousOpponent"]
                     for _, event in rows], dtype=np.int8)
    seed = np.array([s for s, _ in rows], dtype=np.int16)
    action = np.array([event["action"] for _, event in rows], dtype=np.int8)
    rewards = np.array([event["reward"] for _, event in rows], dtype=np.int8)
    target = np.array([event["targetSpikes"] for _, event in rows], dtype=float)
    kc_total = np.array([event["kcSpikes"] for _, event in rows], dtype=float)
    mbon = np.array([event["rates"] for _, event in rows], dtype=float)
    sub = json.loads((ROOT / "data" / "subgraph.json").read_text())
    kc_ids = [i for i, role in enumerate(sub["neurons"]["role"]) if role == "kc"]
    kc_index = {neuron: j for j, neuron in enumerate(kc_ids)}
    kc_pattern = np.zeros((len(rows), len(kc_ids)), dtype=np.float32)
    for row_index, (_, event) in enumerate(rows):
        for neuron, count in event["kcActive"]:
            kc_pattern[row_index, kc_index[neuron]] = count
    groups = seed * 4 + prev
    # The following is an association check within matched seed and visual input,
    # not a decoder used by Fly B.
    def distance(values):
        total, weight = 0.0, 0
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            zero = indices[bits[indices] == 0]
            one = indices[bits[indices] == 1]
            if len(zero) < 3 or len(one) < 3:
                continue
            difference = values[zero].mean(axis=0) - values[one].mean(axis=0)
            total += len(indices) * float(np.linalg.norm(difference))
            weight += len(indices)
        return total / weight if weight else float("nan"), weight / len(bits)
    kc_total_distance, coverage = distance(kc_total[:, None])
    kc_pattern_distance, _ = distance(kc_pattern)
    mbon_distance, _ = distance(mbon)
    return {"rounds": len(rows), "bitZeroFraction": mean(bits == 0),
            "labelCounts": [int((bits == k).sum()) for k in (0, 1)],
            "targetSpikesByBit": [target[bits == k].mean(axis=0).tolist() for k in (0, 1)],
            "kcSpikesByBit": [mean(kc_total[bits == k]) for k in (0, 1)],
            "mbonRatesByBit": [mbon[bits == k].mean(axis=0).tolist() for k in (0, 1)],
            "kcTotalDistanceWithinSeedAndObservation": kc_total_distance,
            "kcPopulationDistanceWithinSeedAndObservation": kc_pattern_distance,
            "mbonRateDistanceWithinSeedAndObservation": mbon_distance,
            "withinObservationCoverage": coverage,
            "actionDistributionByBit": [(np.bincount(action[bits == k], minlength=3) /
                                          max(1, int((bits == k).sum()))).tolist() for k in (0, 1)],
            "actionInformationGivenPreviousMoveAndSeed": information_test(bits, action, groups),
            "rewardInformationGivenPreviousMoveAndSeed": information_test(bits, rewards + 1, groups)}


def paired_probe(seeds, stage="before"):
    by_seed = []
    for seed in seeds:
        suffix = "" if stage == "before" else "_after"
        data = json.loads((ROOT / "data" / f"rps_pretrained_channel_probe_{seed}{suffix}.json").read_text())
        pair = {}
        for row in data["records"]:
            pair.setdefault((row["rep"], row["previousOpponent"]), {})[row["bit"]] = row
        kc_difference, mbon_difference, action_difference = [], [], []
        correct_by_bit = [[], []]
        targets = [[], []]
        for values in pair.values():
            if set(values) != {0, 1}:
                raise RuntimeError("Incomplete paired probe")
            zero, one = values[0], values[1]
            kc_zero = dict(zero["kcActive"])
            kc_one = dict(one["kcActive"])
            kc_difference.append(sum(abs(kc_zero.get(i, 0) - kc_one.get(i, 0))
                                     for i in kc_zero.keys() | kc_one.keys()))
            mbon_difference.append(float(np.linalg.norm(np.array(zero["mbonRates"])
                                                         - np.array(one["mbonRates"]))))
            action_difference.append(int(zero["action"] != one["action"]))
            correct_by_bit[0].append(int(zero["action"] == zero["previousOpponent"]))
            correct_by_bit[1].append(int(one["action"] == one["previousOpponent"]))
            targets[0].append(zero["targetSpikes"])
            targets[1].append(one["targetSpikes"])
        by_seed.append({"seed": seed, "pairedTrials": len(pair),
                        "kcPopulationSpikeCountL1Difference": mean(kc_difference),
                        "mbonRateL2Difference": mean(mbon_difference),
                        "actionChangesFraction": mean(action_difference),
                        "correctActionFractionByBit": [mean(values) for values in correct_by_bit],
                        "targetSpikesByBit": [np.mean(values, axis=0).tolist()
                                              for values in targets]})
    return {"stage": stage, "bySeed": by_seed,
            **{metric: {"mean": mean([row[metric] for row in by_seed]),
                        "bootstrap95": bootstrap_ci([row[metric] for row in by_seed])}
               for metric in ("kcPopulationSpikeCountL1Difference",
                              "mbonRateL2Difference", "actionChangesFraction")}}


def main():
    seeds = [int(value) for value in sys.argv[1:]]
    if len(seeds) < 2:
        raise SystemExit("usage: python tools/analyze_rps_pretrained_teacher.py SEED SEED ...")
    runs = [json.loads((ROOT / "data" / f"rps_pretrained_teacher_{seed}.json").read_text())
            for seed in seeds]
    if [run["seed"] for run in runs] != seeds:
        raise RuntimeError("Evaluation seeds mismatch")
    if any(run["lock"] != runs[0]["lock"] for run in runs):
        raise RuntimeError("Source lock differs between runs")
    result = {"seeds": seeds, "lock": runs[0]["lock"],
              "teacherPretraining": {
                  "newStrategyLast5BySeed": [mean(run["teacherTraining"]["pretrainedRates"][-5:])
                                             for run in runs],
                  "oldStrategyLast5BySeed": [mean(run["teacherTraining"]["simultaneousPretrainRates"][-5:])
                                             for run in runs]},
              "teacherSignals": {kind: teacher_info(runs, kind)
                                 for kind in ("pretrained", "simultaneous", "untrained")},
              "students": {kind: student_metrics(runs, kind) for kind in KINDS},
              "pairedFrozenProbe": {stage: paired_probe(seeds, stage)
                                    for stage in ("before", "after")},
              "studentActivity": {kind: signal_activity(runs, kind)
                                  for kind in KINDS if kind != "alone"}}
    before = result["pairedFrozenProbe"]["before"]["bySeed"]
    after = result["pairedFrozenProbe"]["after"]["bySeed"]
    probe_change = [later["actionChangesFraction"] - earlier["actionChangesFraction"]
                    for earlier, later in zip(before, after)]
    result["pairedFrozenProbe"]["afterMinusBeforeActionChanges"] = {
        "bySeed": probe_change, "mean": mean(probe_change),
        "bootstrap95": bootstrap_ci(probe_change)}
    result["pairedPretrainedMinusControls"] = {}
    for metric in ("early24", "early72", "post240", "frozenDisconnected"):
        trained = [row[metric] for row in result["students"]["pretrained"]["bySeed"]]
        result["pairedPretrainedMinusControls"][metric] = {}
        for kind in KINDS:
            if kind == "pretrained":
                continue
            control = [row[metric] for row in result["students"][kind]["bySeed"]]
            differences = [a - b for a, b in zip(trained, control)]
            result["pairedPretrainedMinusControls"][metric][kind] = {
                "bySeed": differences, "mean": mean(differences),
                "bootstrap95": bootstrap_ci(differences)}
    output = ROOT / "data" / "rps_pretrained_teacher_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for kind in ("pretrained", "simultaneous", "untrained"):
        values = result["teacherSignals"][kind]["bitZeroByEpisode"]
        axes[0].plot(np.arange(1, 11), values, marker="o", lw=1.8,
                     label=kind, color=COLORS[kind])
    axes[0].set(xlabel="Episode after switch", ylabel="Fraction of messages with bit = 0",
                ylim=(-.03, 1.03), title="Actual locked neuron 2132 messages")
    axes[0].legend()
    for kind in KINDS:
        wins = np.array([[event["reward"] == 1 for event in run["conditions"][kind]["postEvents"]]
                         for run in runs], dtype=float)
        rolling = np.convolve(wins.mean(axis=0), np.ones(24) / 24, mode="valid")
        axes[1].plot(np.arange(24, 241), rolling, label=kind, color=COLORS[kind], lw=1.7)
    axes[1].axhline(1 / 3, color="black", ls=":", lw=1, label="chance")
    axes[1].set(xlabel="Rounds after switch (24-round rolling window)",
                ylabel="Fly B win rate", title="Paired independent evaluation seeds")
    axes[1].legend(ncol=3, fontsize=8)
    figure = ROOT / "docs" / "rps-diagnostics" / "pretrained-teacher-focus.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    probe = result["pairedFrozenProbe"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for earlier, later in zip(probe["before"]["bySeed"], probe["after"]["bySeed"]):
        axes[0].plot([0, 1], [earlier["actionChangesFraction"], later["actionChangesFraction"]],
                     marker="o", color="#8a8a8a", alpha=.7)
    axes[0].set(xticks=[0, 1], xticklabels=["Before", "After"], ylim=(0, 1),
                ylabel="Fraction of paired trials with a different action",
                title="Causal effect of the two bit values")
    width = .33
    for stage, position in (("before", 0), ("after", 1)):
        cells = probe[stage]["bySeed"]
        values = np.mean([row["correctActionFractionByBit"] for row in cells], axis=0)
        axes[1].bar(position - width / 2, values[0], width=width,
                    color=COLORS["pretrained"], label="bit 0" if stage == "before" else None)
        axes[1].bar(position + width / 2, values[1], width=width,
                    color=COLORS["simultaneous"], label="bit 1" if stage == "before" else None)
    axes[1].axhline(1 / 3, color="black", ls=":", lw=1)
    axes[1].set(xticks=[0, 1], xticklabels=["Before", "After"], ylim=(0, 1),
                ylabel="Correct action for deterministic new strategy",
                title="Frozen Fly B responses")
    axes[1].legend()
    probe_figure = ROOT / "docs" / "rps-diagnostics" / "pretrained-teacher-probe.png"
    fig.savefig(probe_figure, dpi=160)
    plt.close(fig)
    print(f"saved {output}, {figure}, and {probe_figure}")
    for kind in KINDS:
        cell = result["students"][kind]
        print(kind, "first72", round(cell["early72"]["mean"], 3),
              "frozen", round(cell["frozenDisconnected"]["mean"], 3))


if __name__ == "__main__":
    main()
