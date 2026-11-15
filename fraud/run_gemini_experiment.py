"""Run the predeclared 12-message Gemini study once; see the locked protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import SEED, evaluate, feature_vector, fit_student, make_dataset
from .teacher import teacher_model, teacher_provider
from .train_frontier import OUTPUT as STUDENT_OUTPUT, train

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "fraud" / "gemini-experiment.json"
MODEL = "gemini-3.5-flash-lite"
MESSAGES = 12


def run() -> dict:
    if OUTPUT.exists() or STUDENT_OUTPUT.exists():
        raise FileExistsError("Prospective result or student artifact already exists; refusing to overwrite")
    if teacher_provider() != "gemini" or teacher_model() != MODEL:
        raise RuntimeError(f"Locked protocol requires TEACHER_PROVIDER=gemini and GEMINI_MODEL={MODEL}")

    # train() requests only features from the first 12 unlabeled training cases.
    student = train(MESSAGES, save=False)
    data = make_dataset()
    rows = data["train"]
    order = np.random.default_rng(SEED + 99).permutation(len(rows))
    labeled = set(map(int, order[:72]))
    raw = np.stack([feature_vector(row["features"]) for row in rows])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-8] = 1
    baseline = fit_student(rows, labeled, None, mean, scale)
    assert np.allclose(mean, student["mean"]) and np.allclose(scale, student["scale"])
    held_out = evaluate({"models": {"baseline": baseline, "gemini_guided": np.array(student["weights"])},
                         "mean": mean, "scale": scale}, data["test"])
    base, guided = held_out["baseline"], held_out["gemini_guided"]
    passed = (guided["f1"] > base["f1"] and guided["fraud_recall"] >= base["fraud_recall"]
              and guided["false_positive_rate"] <= base["false_positive_rate"] + .02)
    result = {"protocol": "docs/synapse-fraud-gemini-protocol.md", "teacher_provider": "gemini",
              "teacher_model": MODEL, "teacher_messages": MESSAGES, "true_label_examples": 72,
              "held_out_cases": len(data["test"]), "threshold": .5, "success_gate_passed": passed,
              "held_out": held_out}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    student["evaluation"] = "Separate prospective Gemini experiment; see data/fraud/gemini-experiment.json"
    with STUDENT_OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(student, file, indent=2)
        file.write("\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"teacher_messages": result["teacher_messages"],
        "success_gate_passed": result["success_gate_passed"],
        "metrics": {name: {key: value[key] for key in ("precision", "fraud_recall", "f1", "confusion")}
                    for name, value in result["held_out"].items()}}, indent=2))
