"""Run the locked Synapse Fraud synthetic experiment once and save raw predictions."""
from __future__ import annotations

import json
from pathlib import Path

from .core import FEATURE_NAMES, SEED, evaluate, make_dataset, train_models

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "fraud" / "experiment.json"


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("Write-once experiment already exists")
    data = make_dataset()
    train_ids = {r["account_id"] for r in data["train"]}
    dev_ids = {r["account_id"] for r in data["development"]}
    test_ids = {r["account_id"] for r in data["test"]}
    assert not (train_ids & dev_ids or train_ids & test_ids or dev_ids & test_ids)
    models = train_models(data["train"])
    assert all(any(abs(v) > 1e-6 for v in model) for model in models["models"].values())
    results = evaluate(models, data["test"])
    guided = results["guided"]
    passed = (guided["f1"] > max(results["independent"]["f1"], results["shuffled"]["f1"])
              and all(guided["false_positive_rate"] <= results[n]["false_positive_rate"] + .02
                      for n in ("independent", "shuffled")))
    artifact = {"protocol": "docs/synapse-fraud-protocol.md", "seed": SEED,
        "dataset": {name: {"accounts": len({r["account_id"] for r in rows}), "cases": len(rows),
                            "fraud": sum(r["label"] for r in rows)} for name, rows in data.items()},
        "feature_names": FEATURE_NAMES, "labeled_training_cases": len(models["labeled"]),
        "teacher_messages": models["messages"], "teacher_categories": models["teacher_categories"],
        "teacher_source": "deterministic rule fixture; no frontier API call",
        "evaluation_teacher_disconnected": True, "success_gate_passed": passed,
        "models": {name: [float(x) for x in weights] for name, weights in models["models"].items()},
        "scaler": {"mean": models["mean"].tolist(), "scale": models["scale"].tolist()},
        "held_out": results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")
    return artifact


if __name__ == "__main__":
    result = run()
    print(json.dumps({"dataset": result["dataset"], "success_gate_passed": result["success_gate_passed"],
        "metrics": {name: {key: round(value[key], 3) for key in ("precision", "recall", "f1", "false_positive_rate")}
                    for name, value in result["held_out"].items()}}, indent=2))
