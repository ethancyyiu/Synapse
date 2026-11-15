"""One predeclared merchant-tenure teacher study; see the separate protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import FEATURE_NAMES, SEED, make_dataset, validate_teacher_output
from .teacher import CACHE_DIR, gemini_teacher, teacher_model, teacher_provider

ROOT = Path(__file__).resolve().parent.parent
DEVELOPMENT = ROOT / "data" / "fraud" / "tenure-development.json"
HELD_OUT = ROOT / "data" / "fraud" / "tenure-heldout.json"
FEATURES = FEATURE_NAMES + ("merchant_older_count",)
MODEL = "gemini-3.5-flash-lite"
TENURE_PROMPT = ("Classify this synthetic purchase for human review using only the seven supplied "
    "history features. A burst at a merchant with no purchases older than 24 hours can be more "
    "concerning than a similar burst at an established merchant; consider amount and category "
    "context too. Return LOW, REVIEW, or HIGH with a brief explanation, numeric-feature-grounded "
    "indicators, and uncertainty. These are review priorities, not verified fraud labels. "
    "Do not invent account, merchant, or customer facts.")
TARGETS = {"LOW": .05, "REVIEW": .5, "HIGH": .95}


def extended_features(row: dict) -> dict:
    transaction = row["transaction"]
    when = datetime.fromisoformat(transaction["date"].replace("Z", "+00:00"))
    older = sum(p["merchant_id"] == transaction["merchant_id"] and
                datetime.fromisoformat(p["date"].replace("Z", "+00:00")) < when - timedelta(hours=24)
                for p in row["history"])
    return {**row["features"], "merchant_older_count": older}


def vector(f: dict) -> np.ndarray:
    return np.array([f[name] for name in FEATURES], dtype=float)


def selected_indices(rows: list[dict], labeled: set[int]) -> list[int]:
    groups = {False: [], True: []}
    for i, row in enumerate(rows):
        if i in labeled:
            continue
        f = extended_features(row)
        if f["recent_count"] >= 2 and f["same_merchant_count"] >= 2:
            established = f["merchant_older_count"] > 0
            if len(groups[established]) < 6:
                groups[established].append(i)
    chosen = groups[False] + groups[True]
    if len(groups[False]) != 6 or len(groups[True]) != 6:
        raise RuntimeError("Insufficient unlabeled burst examples for locked selection")
    return chosen


def cached_teacher(f: dict) -> dict:
    limited = {name: round(float(f[name]), 3) for name in FEATURES}
    digest = hashlib.sha256(json.dumps(["gemini-tenure-v1", MODEL, TENURE_PROMPT, limited],
                                    sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{digest}.json"
    if path.exists():
        return validate_teacher_output(json.loads(path.read_text(encoding="utf-8")))
    result = gemini_teacher(f, feature_names=FEATURES, prompt=TENURE_PROMPT)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def fit(rows: list[dict], labeled: set[int], targets: dict[int, float], mean: np.ndarray,
        scale: np.ndarray) -> np.ndarray:
    chosen = [i for i in range(len(rows)) if i in labeled or i in targets]
    x = np.stack([(vector(extended_features(rows[i])) - mean) / scale for i in chosen])
    x = np.c_[np.ones(len(x)), x]
    y = np.array([rows[i]["label"] if i in labeled else targets[i] for i in chosen], dtype=float)
    weights = np.array([1.0 if i in labeled else .4 for i in chosen])
    w = np.zeros(len(FEATURES) + 1)
    for _ in range(400):
        pred = 1 / (1 + np.exp(-np.clip(x @ w, -25, 25)))
        grad = x.T @ ((pred - y) * weights) / weights.sum()
        grad[1:] += .01 * w[1:]
        w -= .12 * grad
    return w


def evaluate(rows: list[dict], models: dict[str, np.ndarray], mean: np.ndarray,
             scale: np.ndarray) -> dict:
    results = {}
    for name, weights in models.items():
        predictions = []
        for row in rows:
            x = (vector(extended_features(row)) - mean) / scale
            value = float(1 / (1 + np.exp(-np.clip(weights[0] + x @ weights[1:], -25, 25))))
            predictions.append({"id": row["transaction"]["id"], "label": row["label"],
                                "score": value, "prediction": int(value >= .5)})
        tp = sum(p["label"] == 1 and p["prediction"] == 1 for p in predictions)
        fp = sum(p["label"] == 0 and p["prediction"] == 1 for p in predictions)
        fn = sum(p["label"] == 1 and p["prediction"] == 0 for p in predictions)
        tn = sum(p["label"] == 0 and p["prediction"] == 0 for p in predictions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        results[name] = {"confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                         "fraud_recall": recall, "precision": precision,
                         "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                         "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                         "predictions": predictions}
    return results


def gate(results: dict) -> bool:
    control, teacher = results["matched_control"], results["gemini_teacher"]
    return (teacher["f1"] > control["f1"] and
            teacher["fraud_recall"] >= control["fraud_recall"] and
            teacher["false_positive_rate"] <= control["false_positive_rate"] + .02)


def run_development() -> dict:
    if DEVELOPMENT.exists() or HELD_OUT.exists():
        raise FileExistsError("Separate tenure study already exists; refusing to overwrite")
    if teacher_provider() != "gemini" or teacher_model() != MODEL:
        raise RuntimeError("Locked tenure study requires Gemini 3.5 Flash-Lite")
    data = make_dataset(range(60))
    rows = data["train"]
    order = np.random.default_rng(SEED + 99).permutation(len(rows))
    labeled = set(map(int, order[:72]))
    chosen = selected_indices(rows, labeled)
    raw = np.stack([vector(extended_features(row)) for row in rows])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-8] = 1
    targets = {}
    risk_counts = {risk: 0 for risk in TARGETS}
    for i in chosen:
        risk = cached_teacher(extended_features(rows[i]))["risk"]
        risk_counts[risk] += 1
        targets[i] = TARGETS[risk]
    models = {"matched_control": fit(rows, labeled, {}, mean, scale),
              "gemini_teacher": fit(rows, labeled, targets, mean, scale)}
    results = evaluate(data["development"], models, mean, scale)
    artifact = {"protocol": "docs/synapse-fraud-tenure-protocol.md", "teacher_model": MODEL,
                "teacher_messages": len(chosen), "teacher_risk_counts": risk_counts,
                "true_label_examples": len(labeled), "features": FEATURES, "threshold": .5,
                "mean": mean.tolist(), "scale": scale.tolist(),
                "weights": {name: w.tolist() for name, w in models.items()},
                "development_gate_passed": gate(results), "development": results}
    DEVELOPMENT.parent.mkdir(parents=True, exist_ok=True)
    with DEVELOPMENT.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")
    return artifact


def run_final() -> dict:
    if HELD_OUT.exists():
        raise FileExistsError("New held-out result already exists; refusing to overwrite")
    saved = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if saved["features"] != list(FEATURES) or saved["teacher_messages"] != 12:
        raise RuntimeError("Development artifact does not match locked procedure")
    models = {name: np.array(w) for name, w in saved["weights"].items()}
    # Generate the previously uninspected accounts only after weights are frozen.
    rows = make_dataset(range(72, 84))["test"]
    results = evaluate(rows, models, np.array(saved["mean"]), np.array(saved["scale"]))
    artifact = {"protocol": saved["protocol"], "teacher_model": MODEL,
                "teacher_messages": saved["teacher_messages"], "held_out_cases": len(rows),
                "threshold": .5, "success_gate_passed": gate(results), "held_out": results}
    with HELD_OUT.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "final"))
    args = parser.parse_args()
    result = run_development() if args.phase == "development" else run_final()
    scores = result["development"] if args.phase == "development" else result["held_out"]
    print(json.dumps({"phase": args.phase, "teacher_messages": result["teacher_messages"],
                      "gate_passed": result["development_gate_passed"] if args.phase == "development" else result["success_gate_passed"],
                      "metrics": {name: {key: values[key] for key in ("fraud_recall", "precision", "f1", "false_positive_rate", "confusion")}
                                  for name, values in scores.items()}}, indent=2))
