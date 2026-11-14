"""Optional bounded OpenAI teacher training; never changes the locked fixture study.

Run: python -m fraud.train_frontier --teacher-examples 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .core import FEATURE_NAMES, SEED, feature_vector, fit_student, make_dataset
from .teacher import DEFAULT_MODEL, cached_frontier_teacher, frontier_available

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "fraud" / "frontier-student.json"
TARGETS = {"LOW": .05, "REVIEW": .5, "HIGH": .95}


def train(teacher_examples: int = 20, save: bool = True) -> dict:
    if not frontier_available():
        raise RuntimeError("Set OPENAI_API_KEY before training a frontier-guided student")
    data = make_dataset()
    rows = data["train"]
    if not 1 <= teacher_examples <= len(rows) - round(.15 * len(rows)):
        raise ValueError("teacher_examples must be between 1 and 408")
    rng = np.random.default_rng(SEED + 99)
    order = rng.permutation(len(rows))
    labeled = set(map(int, order[:round(.15 * len(rows))]))
    candidates = list(map(int, order[round(.15 * len(rows)):round(.15 * len(rows)) + teacher_examples]))
    raw = np.stack([feature_vector(row["features"]) for row in rows])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-8] = 1
    targets = np.full(len(rows), np.nan)
    for i in candidates:
        targets[i] = TARGETS[cached_frontier_teacher(rows[i]["features"])["risk"]]
    weights = fit_student(rows, labeled, targets, mean, scale)
    artifact = {"kind": "optional frontier-guided demo model", "seed": SEED,
        "teacher_model": os.environ.get("SYNAPSE_FRAUD_TEACHER_MODEL", DEFAULT_MODEL),
        "teacher_examples": len(candidates), "true_label_examples": len(labeled),
        "feature_names": FEATURE_NAMES, "weights": weights.tolist(),
        "mean": mean.tolist(), "scale": scale.tolist(),
        "evaluation": "No held-out result; distinct from the locked fixture experiment"}
    if save:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        temporary.replace(OUTPUT)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-examples", type=int, default=20,
                        help="number of unlabeled training cases sent to the teacher (default 20; max 408)")
    args = parser.parse_args()
    result = train(args.teacher_examples)
    print(f"Saved local student trained with {result['teacher_examples']} frontier messages to {OUTPUT}")
