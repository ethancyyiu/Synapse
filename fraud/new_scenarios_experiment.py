"""One locked three-arm synthetic Gemini study; protocol in docs/."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .core import features, validate_teacher_output
from .teacher import CACHE_DIR, gemini_teacher, teacher_model, teacher_provider
from .tenure_experiment import FEATURES, extended_features, evaluate, vector

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "docs" / "synapse-fraud-new-scenarios-protocol.md"
DEVELOPMENT = ROOT / "data" / "fraud" / "new-scenarios-development.json"
HELDOUT = ROOT / "data" / "fraud" / "new-scenarios-heldout.json"
MODEL = "gemini-3.5-flash-lite"
SEED = 20261114
TRAIN_FAMILIES = ("marketplace_probe", "service_charge", "routine_disruption")
DEV_FAMILIES = ("vendor_cluster", "renewal_jump")
TEST_FAMILIES = ("ticket_exchange", "pharmacy_errand")
NEW_SCENARIO_PROMPT = (
    "Assign LOW, REVIEW, or HIGH review priority to this synthetic transaction using only seven "
    "numeric history features. Return the validated JSON risk, explanation, indicators, and uncertainty. "
    "Give at most four indicators, each at most 55 characters. Each risk-increasing indicator "
    "must begin with an exact supplied feature name followed by a colon and a short numeric reason. "
    "Indicators are evidence, not a verified fraud label. "
    "Do not infer an account, customer, merchant, or scenario identity."
)
TARGETS = {"LOW": 0.05, "REVIEW": 0.5, "HIGH": 0.95}


def transaction(account: int, tag: str, merchant: str, category: str, amount: float,
                when: datetime) -> dict:
    return {"id": f"new-{account}-{tag}", "account_id": f"new-{account}",
            "merchant_id": f"new-{account}-{merchant}", "merchant_name": merchant,
            "category": category, "amount": round(float(amount), 2),
            "date": when.isoformat().replace("+00:00", "Z")}


def make_row(account: int, family: str, family_index: int) -> dict:
    rng = np.random.default_rng(SEED + 37 * account + 701 * family_index)
    intent = int((account + family_index) % 2)  # independent synthetic ground truth
    typical = float(rng.uniform(28, 85))
    when = datetime(2026, 8, 1, 12, tzinfo=timezone.utc) + timedelta(hours=account % 13)
    categories = ("groceries", "utilities", "dining", "pharmacy")
    history = [transaction(account, f"h{i}", f"established-{i % 4}", categories[i % 4],
                           typical * rng.uniform(0.7, 1.3), when - timedelta(days=32 - 2 * i))
               for i in range(10)]

    def recent(tag: str, merchant: str, category: str, multiplier: float, hours: float) -> None:
        history.append(transaction(account, tag, merchant, category,
                                   typical * multiplier * rng.uniform(.85, 1.15),
                                   when - timedelta(hours=hours)))

    candidate_merchant, candidate_category, multiplier = "vendor-new", "groceries", 1.4
    if family in {"marketplace_probe", "vendor_cluster"}:
        count = 3 if family == "marketplace_probe" else 4
        for i in range(count):
            recent(f"r{i}", f"vendor-{i}" if intent else f"established-{i % 4}",
                   categories[i % 4], .35 if family == "marketplace_probe" else .8, i + 1)
        multiplier = 1.4 if family == "marketplace_probe" else 1.1
    elif family in {"service_charge", "renewal_jump"}:
        candidate_merchant, candidate_category = "established-1", "utilities"
        for i in range(2 if intent else 1):
            recent(f"r{i}", candidate_merchant, candidate_category, 1.0, i + 1)
        multiplier = 1.8 if family == "service_charge" else 2.3
    elif family == "routine_disruption":
        for i in range(2):
            recent(f"r{i}", f"outside-{i}" if intent else f"established-{i}",
                   "travel" if intent else categories[i], 1.2, i + 2)
        multiplier = 2.0
    elif family == "ticket_exchange":
        candidate_merchant = "ticket-new" if intent else "established-2"
        candidate_category = "events" if intent else "dining"
        for i in range(2):
            recent(f"r{i}", candidate_merchant, candidate_category, 1.3, i + 1)
        recent("other", "event-vendor", "events", .5, 4)
        multiplier = 2.5
    elif family == "pharmacy_errand":
        candidate_merchant = "pharmacy-new" if intent else "established-3"
        candidate_category = "health" if intent else "pharmacy"
        for i in range(3):
            recent(f"r{i}", f"health-{i}" if intent else "established-3",
                   candidate_category, .8, i + 1)
        multiplier = 1.0
    else:
        raise ValueError("Unknown scenario family")

    candidate = transaction(account, "candidate", candidate_merchant, candidate_category,
                            typical * multiplier * rng.uniform(.85, 1.15), when)
    return {"account_id": f"new-{account}", "transaction": candidate, "history": history,
            "features": features(candidate, history), "label": intent, "scenario": family}


def generate(accounts: range, families: tuple[str, ...]) -> list[dict]:
    return [make_row(account, family, i) for account in accounts
            for i, family in enumerate(families)]


def fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes() + PROTOCOL.read_bytes()).hexdigest()


def cached_teacher(f: dict) -> tuple[dict, str]:
    limited = {name: round(float(f[name]), 3) for name in FEATURES}
    digest = hashlib.sha256(json.dumps(["new-scenarios-v1", MODEL, NEW_SCENARIO_PROMPT, limited],
                                    sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{digest}.json"
    if path.exists():
        return validate_teacher_output(json.loads(path.read_text(encoding="utf-8"))), "cached"
    result = gemini_teacher(f, feature_names=FEATURES, prompt=NEW_SCENARIO_PROMPT)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result, "live"


def cited_features(message: dict) -> list[str]:
    return sorted({name for indicator in message["indicators"] for name in FEATURES
                   if indicator.startswith(name + ":")})


def fit(rows: list[dict], base: set[int], extra: set[int], targets: dict[int, float],
        extra_weights: dict[int, float], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    chosen = sorted(base | extra)
    x = np.stack([(vector(extended_features(rows[i])) - mean) / scale for i in chosen])
    x = np.c_[np.ones(len(x)), x]
    y = np.array([rows[i]["label"] if i in base else targets[i] for i in chosen], dtype=float)
    weights = np.array([1.0 if i in base else extra_weights[i] for i in chosen])
    w = np.zeros(len(FEATURES) + 1)
    for _ in range(400):
        pred = 1 / (1 + np.exp(-np.clip(x @ w, -25, 25)))
        grad = x.T @ ((pred - y) * weights) / weights.sum()
        grad[1:] += .01 * w[1:]
        w -= .12 * grad
    return w


def gate(results: dict) -> bool:
    teacher = results["gemini_guided"]
    controls = (results["limited_baseline"], results["independent_labels"])
    return all(teacher["f1"] > c["f1"] and teacher["fraud_recall"] >= c["fraud_recall"]
               and teacher["false_positive_rate"] <= c["false_positive_rate"] + .05
               for c in controls)


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


def run_development() -> dict:
    if DEVELOPMENT.exists() or HELDOUT.exists():
        raise FileExistsError("New-scenario study already saved; refusing to overwrite")
    if teacher_provider() != "gemini" or teacher_model() != MODEL:
        raise RuntimeError("Locked study requires Gemini 3.5 Flash-Lite")
    rows = generate(range(1000, 1036), TRAIN_FAMILIES)
    permutation = np.random.default_rng(SEED + 1).permutation(len(rows))
    base = set(map(int, permutation[:36]))
    extra = set(map(int, permutation[36:48]))
    raw = np.stack([vector(extended_features(row)) for row in rows])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale[scale < 1e-8] = 1
    teacher_targets, weights, messages = {}, {}, []
    for i in sorted(extra):
        f = extended_features(rows[i])
        message, source = cached_teacher(f)
        cited = cited_features(message)
        teacher_targets[i] = TARGETS[message["risk"]]
        weights[i] = .2 + .1 * min(len(cited), 4)
        messages.append({"training_id": rows[i]["transaction"]["id"], "risk": message["risk"],
                         "indicators": message["indicators"], "grounded_features": cited,
                         "source": source})
    models = {
        "limited_baseline": fit(rows, base, set(), {}, {}, mean, scale),
        "independent_labels": fit(rows, base, extra, {i: rows[i]["label"] for i in extra},
                                  {i: .4 for i in extra}, mean, scale),
        "gemini_guided": fit(rows, base, extra, teacher_targets, weights, mean, scale),
    }
    development = evaluate(generate(range(2000, 2012), DEV_FAMILIES), models, mean, scale)
    artifact = {"protocol": "docs/synapse-fraud-new-scenarios-protocol.md", "fingerprint": fingerprint(),
                "model": MODEL, "features": FEATURES, "threshold": .5,
                "splits": {"training_accounts": [1000, 1035], "development_accounts": [2000, 2011],
                           "heldout_accounts": [3000, 3023], "training_families": TRAIN_FAMILIES,
                           "development_families": DEV_FAMILIES, "heldout_families": TEST_FAMILIES},
                "base_label_count": 36, "extra_label_count": 12, "teacher_message_count": len(messages),
                "teacher_messages": messages, "base_indices": sorted(base), "extra_indices": sorted(extra),
                "mean": mean.tolist(), "scale": scale.tolist(),
                "weights": {name: model.tolist() for name, model in models.items()},
                "development_gate_passed": gate(development), "development": development}
    save(DEVELOPMENT, artifact)
    return artifact


def run_final() -> dict:
    if HELDOUT.exists():
        raise FileExistsError("Final new-scenario result already saved; refusing to overwrite")
    saved = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    if saved["fingerprint"] != fingerprint() or saved["teacher_message_count"] != 12:
        raise RuntimeError("Protocol or implementation changed since development")
    models = {name: np.array(weights) for name, weights in saved["weights"].items()}
    results = evaluate(generate(range(3000, 3024), TEST_FAMILIES), models,
                       np.array(saved["mean"]), np.array(saved["scale"]))
    artifact = {"protocol": saved["protocol"], "fingerprint": saved["fingerprint"],
                "model": MODEL, "teacher_message_count": 12, "heldout_cases": 48,
                "threshold": .5, "success_gate_passed": gate(results), "held_out": results}
    save(HELDOUT, artifact)
    return artifact


if __name__ == "__main__":
    import sys
    result = run_development() if sys.argv[1:] == ["development"] else (
        run_final() if sys.argv[1:] == ["final"] else None)
    if result is None:
        raise SystemExit("Use development or final")
    key = "development" if "development" in result else "held_out"
    print(json.dumps({"gate": result.get("development_gate_passed", result.get("success_gate_passed")),
                      "teacher_messages": result["teacher_message_count"],
                      "results": {name: {k: arm[k] for k in ("fraud_recall", "precision", "f1",
                                                           "false_positive_rate", "confusion")}
                                  for name, arm in result[key].items()}}, indent=2))
