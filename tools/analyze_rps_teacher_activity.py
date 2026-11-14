"""Rank presynaptic RPS activity on calibration seeds only.

Usage: python tools/analyze_rps_teacher_activity.py calibration 41 42 43
       python tools/analyze_rps_teacher_activity.py evaluation 61 62 63
The evaluation command reports locked features; it never changes the lock.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"mid", "mbon", "dn", "pam", "ppl1"}
THRESHOLDS = ("spike>0", "trace>=0.5", "trace>=1.5")


def load(split, seed, variant="real"):
    suffix = "" if variant == "real" else f"_{variant}"
    path = ROOT / "data" / f"rps_teacher_activity_{split}_{seed}{suffix}.json.gz"
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    n = len(data["neuronIds"])
    out = {}
    for condition in ("trained", "untrained"):
        rows = data[condition]["samples"]
        counts = np.zeros((len(rows), n), dtype=np.uint8)
        traces = np.zeros((len(rows), n), dtype=np.float32)
        for r, row in enumerate(rows):
            for j, value in row["spikes"]:
                counts[r, j] = value
            for j, value in row["traces"]:
                traces[r, j] = value
        out[condition] = {
            "X": np.concatenate((counts > 0, traces >= .5, traces >= 1.5), axis=1),
            "prev": np.array([3 if row["previousOpponent"] is None else row["previousOpponent"]
                              for row in rows], dtype=np.int8),
            "next": np.array([row["nextOpponent"] for row in rows], dtype=np.int8),
            "action": np.array([row["action"] for row in rows], dtype=np.int8),
            "reward": np.array([3 if row["previousReward"] is None else row["previousReward"] + 1
                                for row in rows], dtype=np.int8),
            "phase": np.array([int(row["episode"] >= 10) for row in rows], dtype=np.int8),
            "episode": np.array([row["episode"] for row in rows], dtype=np.int8),
        }
    return data["neuronIds"], data["neuronRoles"], out


def conditional_information(X, y, context):
    """Plug-in conditional mutual information, in bits, for binary feature columns."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    context = np.asarray(context)
    n, features = X.shape
    result = np.zeros(features)
    k = int(y.max()) + 1
    for group in np.unique(context):
        mask = context == group
        subset = X[mask]
        labels = y[mask]
        size = len(labels)
        onehot = np.eye(k, dtype=np.float32)[labels]
        one = subset.T @ onehot
        label_totals = onehot.sum(axis=0)[None, :]
        zero = label_totals - one
        total_one = one.sum(axis=1)[:, None]
        total_zero = size - total_one
        for joint, x_total in ((one, total_one), (zero, total_zero)):
            with np.errstate(divide="ignore", invalid="ignore"):
                term = joint / n * np.log2(joint * size / (x_total * label_totals))
            result += np.where(joint > 0, np.nan_to_num(term), 0).sum(axis=1)
    return result


def summarize(part):
    X, prev = part["X"], part["prev"]
    next_move, phase = part["next"], part["phase"]
    context_phase = prev * 2 + phase
    return {
        "next_given_prev": conditional_information(X, next_move, prev),
        "next_given_prev_phase": conditional_information(X, next_move, context_phase),
        "phase_given_prev": conditional_information(X, phase, prev),
        "action_given_prev": conditional_information(X, part["action"], prev),
        "reward_history_given_prev": conditional_information(X, part["reward"], prev),
        "previous_move": conditional_information(X, prev, np.zeros(len(prev), dtype=np.int8)),
        "frequency": X.mean(axis=0),
    }


def combine(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def main():
    split = sys.argv[1]
    variant = "shuffled" if "--shuffled" in sys.argv else "real"
    seeds = [int(x) for x in sys.argv[2:] if x != "--shuffled"]
    if split not in {"calibration", "evaluation"} or not seeds:
        raise SystemExit("usage: python tools/analyze_rps_teacher_activity.py calibration|evaluation SEED...")
    loaded = [load(split, seed, variant) for seed in seeds]
    neuron_ids, roles = loaded[0][:2]
    if any(item[0] != neuron_ids for item in loaded):
        raise RuntimeError("Neuron indexing differs between recordings")
    combined = {condition: combine([item[2][condition] for item in loaded])
                for condition in ("trained", "untrained")}
    stats = {condition: summarize(part) for condition, part in combined.items()}
    per_seed = [{condition: summarize(item[2][condition]) for condition in ("trained", "untrained")}
                for item in loaded]
    n = len(neuron_ids)
    eligible = np.array([role in ROLES for role in roles] * len(THRESHOLDS))
    freq = stats["trained"]["frequency"]
    eligible &= (freq >= .1) & (freq <= .9)
    gain = stats["trained"]["next_given_prev"] - stats["untrained"]["next_given_prev"]
    gain_each_seed = np.stack([seed_stats["trained"]["next_given_prev"]
                               - seed_stats["untrained"]["next_given_prev"]
                               for seed_stats in per_seed])
    ranking_eligible = eligible & np.all(gain_each_seed > 0, axis=0)
    rank = np.where(ranking_eligible)[0]
    rank = rank[np.argsort(-gain[rank])]

    def entry(j):
        local = int(j % n)
        metrics = {}
        for name in stats["trained"]:
            metrics[name] = {condition: round(float(stats[condition][name][j]), 6)
                             for condition in ("trained", "untrained")}
        metrics["perSeedNextGain"] = [round(float(seed_stats["trained"]["next_given_prev"][j]
                                                     - seed_stats["untrained"]["next_given_prev"][j]), 6)
                                      for seed_stats in per_seed]
        return {"neuron": neuron_ids[local], "role": roles[local],
                "feature": THRESHOLDS[j // n], "gainBits": round(float(gain[j]), 6),
                "metrics": metrics}

    result = {"split": split, "variant": variant, "seeds": seeds,
              "roundsPerCondition": len(combined["trained"]["prev"]),
              "candidateNeurons": sum(role in ROLES for role in roles),
              "eligibleFeatures": int(eligible.sum()),
              "rankingEligibleFeatures": int(ranking_eligible.sum()),
              "ranking": [entry(j) for j in rank[:30]] if split == "calibration" and variant == "real" else []}
    if split == "calibration" and variant == "real":
        result["associationLeaders"] = {}
        for name in ("previous_move", "action_given_prev", "reward_history_given_prev",
                     "phase_given_prev"):
            order = np.where(eligible)[0]
            order = order[np.argsort(-stats["trained"][name][order])]
            result["associationLeaders"][name] = [entry(j) for j in order[:5]]
    if split == "evaluation" or variant == "shuffled":
        lock = json.loads((ROOT / "data" / "rps_teacher_source_lock.json").read_text())
        j = THRESHOLDS.index(lock["feature"]) * n + neuron_ids.index(lock["neuron"])
        result["lockedSource"] = entry(j)
    if split == "evaluation":
        random = np.random.default_rng(20261101)
        result["lockedSourcePermutation"] = {}
        for condition, part in combined.items():
            signal = part["X"][:, [j]].copy()
            observed = conditional_information(signal, part["next"], part["prev"])[0]
            null = []
            for _ in range(500):
                shuffled = signal.copy()
                for context in np.unique(part["prev"]):
                    group = np.flatnonzero(part["prev"] == context)
                    shuffled[group, 0] = random.permutation(shuffled[group, 0])
                null.append(conditional_information(shuffled, part["next"], part["prev"])[0])
            result["lockedSourcePermutation"][condition] = {
                "observedBits": round(float(observed), 6),
                "null95Bits": round(float(np.quantile(null, .95)), 6),
                "permutationP": round(float((1 + sum(value >= observed for value in null)) / 501), 6),
            }
        result["lockedSourceEpisodeFrequency"] = {
            condition: [round(float(part["X"][part["episode"] == ep, j].mean()), 3)
                        for ep in range(20)] for condition, part in combined.items()}
    suffix = "" if variant == "real" else f"_{variant}"
    output = ROOT / "data" / f"rps_teacher_activity_{split}{suffix}_analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"ranking", "associationLeaders"}}, indent=2))
    if split == "calibration" and variant == "real":
        print("Top candidates:")
        for item in result["ranking"][:10]:
            print(item["neuron"], item["role"], item["feature"], item["gainBits"],
                  item["metrics"]["perSeedNextGain"])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
