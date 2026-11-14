"""Synthetic fraud benchmark, history features, deterministic teacher and local learner."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import time

import numpy as np

SEED = 720260
CATEGORIES = ("groceries", "fuel", "dining", "travel", "electronics", "utilities", "pharmacy")
FEATURE_NAMES = ("log_amount_ratio", "new_merchant", "new_category", "recent_count", "same_merchant_count", "recent_spend_ratio")
RISKS = ("LOW", "REVIEW", "HIGH")
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def stamp(day: int, hour: float = 12) -> str:
    return (START + timedelta(days=day, hours=hour)).isoformat().replace("+00:00", "Z")


def normalize_purchase(raw: dict, merchants: dict | None = None, account_id: str | None = None) -> dict:
    """Normalize a Nessie purchase or a fixture without inferring a fraud label."""
    merchants = merchants or {}
    embedded = raw.get("merchant") if isinstance(raw.get("merchant"), dict) else {}
    merchant_id = str(raw.get("merchant_id") or raw.get("merchantId") or embedded.get("_id") or "unknown")
    merchant = merchants.get(merchant_id, {})
    amount = (float(raw["amount_cents"]) / 100 if "amount_cents" in raw else
              float(raw.get("amount", 0)))
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Invalid purchase amount")
    date = str(raw.get("purchase_date") or raw.get("created_at") or raw.get("date") or "")
    if not date:
        raise ValueError("Purchase lacks a date")
    parsed = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    raw_category = merchant.get("category") or raw.get("category") or "unknown"
    category = raw_category[0] if isinstance(raw_category, list) and raw_category else raw_category
    return {
        "id": str(raw.get("_id") or raw.get("id") or f"{account_id or 'account'}-{merchant_id}-{int(parsed.timestamp())}"),
        "account_id": str(account_id or raw.get("account_id") or "unknown"),
        "amount": round(amount, 2), "date": parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "merchant_id": merchant_id, "merchant_name": str(merchant.get("name") or raw.get("merchant_name") or merchant_id),
        "category": str(category).lower(),
    }


def features(transaction: dict, history: list[dict]) -> dict:
    """All features are computed from purchases earlier than the candidate."""
    when = datetime.fromisoformat(transaction["date"].replace("Z", "+00:00"))
    prior = [p for p in history if datetime.fromisoformat(p["date"].replace("Z", "+00:00")) < when]
    amounts = [p["amount"] for p in prior if p["amount"] > 0]
    median = float(np.median(amounts)) if amounts else max(transaction["amount"], 1.0)
    recent = [p for p in prior if when - datetime.fromisoformat(p["date"].replace("Z", "+00:00")) <= timedelta(hours=24)]
    ratio = transaction["amount"] / max(median, 1)
    return {
        "log_amount_ratio": math.log1p(ratio),
        "new_merchant": int(not any(p["merchant_id"] == transaction["merchant_id"] for p in prior)),
        "new_category": int(not any(p["category"] == transaction["category"] for p in prior)),
        "recent_count": len(recent),
        "same_merchant_count": sum(p["merchant_id"] == transaction["merchant_id"] for p in recent),
        "recent_spend_ratio": sum(p["amount"] for p in recent) / max(median, 1),
        "history_median": round(median, 2),
        "history_size": len(prior),
    }


def feature_vector(f: dict) -> np.ndarray:
    return np.array([f[name] for name in FEATURE_NAMES], dtype=float)


def validate_teacher_output(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("risk") not in RISKS:
        raise ValueError("Teacher must return LOW, REVIEW or HIGH")
    if not isinstance(value.get("explanation"), str) or not 1 <= len(value["explanation"]) <= 300:
        raise ValueError("Invalid teacher explanation")
    indicators = value.get("indicators")
    if not isinstance(indicators, list) or len(indicators) > 6 or any(not isinstance(x, str) or len(x) > 70 for x in indicators):
        raise ValueError("Invalid teacher indicators")
    uncertainty = value.get("uncertainty")
    if uncertainty is not None and (not isinstance(uncertainty, str) or len(uncertainty) > 140):
        raise ValueError("Invalid teacher uncertainty")
    return {"risk": value["risk"], "explanation": value["explanation"],
            "indicators": indicators, "uncertainty": uncertainty}


def fixture_teacher(f: dict) -> dict:
    """A fixed rule teacher that sees features only, never generator labels."""
    indicators = []
    score = 0
    if f["log_amount_ratio"] > math.log1p(3):
        score += 2; indicators.append("Amount exceeds three times historical median")
    if f["new_merchant"]:
        score += 1; indicators.append("Merchant absent from account history")
    if f["new_category"]:
        score += 1; indicators.append("Category absent from account history")
    if f["recent_count"] >= 2:
        score += 1; indicators.append("Several purchases in the prior 24 hours")
    if f["same_merchant_count"] >= 2:
        score += 1; indicators.append("Repeated purchases at this merchant")
    if f["recent_spend_ratio"] > 2.5:
        score += 1; indicators.append("Recent spend is high against account baseline")
    risk = "HIGH" if score >= 4 else "REVIEW" if score >= 2 else "LOW"
    return validate_teacher_output({"risk": risk,
        "explanation": "Multiple history-based indicators warrant review." if risk == "HIGH" else
                       "Some activity differs from history; check context." if risk == "REVIEW" else
                       "No strong history-based risk pattern found.",
        "indicators": indicators, "uncertainty": "Rule-based fixture; this is not a calibrated probability."})


def make_dataset() -> dict[str, list[dict]]:
    splits = {"train": [], "development": [], "test": []}
    scenarios = ("ordinary", "ordinary", "ordinary", "ordinary", "large_known_benign",
                 "new_small_benign", "known_burst_benign", "large_new_fraud", "repeat_new_fraud", "unfamiliar_burst_fraud")
    for account in range(72):
        rng = np.random.default_rng(SEED + account * 1009)
        split = "train" if account < 48 else "development" if account < 60 else "test"
        account_id = f"mock-{account:03d}"
        typical = float(rng.uniform(24, 110))
        familiar = list(rng.choice(CATEGORIES, size=3, replace=False))
        normal_merchants = {cat: f"{account_id}-{cat}-regular" for cat in familiar}
        history = []
        for j in range(30):
            cat = familiar[j % 3]
            amount = round(float(typical * rng.uniform(.5, 1.6)), 2)
            history.append({"id": f"{account_id}-h{j}", "account_id": account_id, "amount": amount,
                            "date": stamp(j, float(rng.uniform(8, 16))), "merchant_id": normal_merchants[cat],
                            "merchant_name": f"Regular {cat.title()}", "category": cat})
        unfamiliar = next(cat for cat in CATEGORIES if cat not in familiar)
        for j, scenario in enumerate(scenarios):
            candidate_time = START + timedelta(days=40 + j * 3, hours=12)
            prior = list(history)
            category = familiar[j % 3]
            merchant_id = normal_merchants[category]
            amount = round(float(typical * rng.uniform(.6, 1.55)), 2)
            label = int(scenario.endswith("_fraud"))
            if scenario == "large_known_benign": amount = round(float(typical * rng.uniform(4.2, 7)), 2)
            if scenario == "new_small_benign":
                merchant_id = f"{account_id}-new-{j}"; amount = round(float(typical * rng.uniform(.5, 1)), 2)
            if scenario == "known_burst_benign":
                for k in range(3):
                    prior.append({"id": f"{account_id}-p{j}-{k}", "account_id": account_id,
                        "amount": round(float(typical * rng.uniform(.7, 1.2)), 2),
                        "date": (candidate_time - timedelta(minutes=15 * (k + 1))).isoformat().replace("+00:00", "Z"),
                        "merchant_id": merchant_id, "merchant_name": f"Regular {category.title()}", "category": category})
            if scenario == "large_new_fraud":
                category = unfamiliar; merchant_id = f"{account_id}-unfamiliar-{j}"
                amount = round(float(typical * rng.uniform(4.5, 8)), 2)
            if scenario == "repeat_new_fraud":
                merchant_id = f"{account_id}-new-{j}"
                for k in range(3):
                    prior.append({"id": f"{account_id}-p{j}-{k}", "account_id": account_id,
                        "amount": round(float(typical * rng.uniform(.8, 1.4)), 2),
                        "date": (candidate_time - timedelta(minutes=12 * (k + 1))).isoformat().replace("+00:00", "Z"),
                        "merchant_id": merchant_id, "merchant_name": "New merchant", "category": category})
            if scenario == "unfamiliar_burst_fraud":
                category = unfamiliar; merchant_id = f"{account_id}-unfamiliar-{j}"
                for k in range(4):
                    prior.append({"id": f"{account_id}-p{j}-{k}", "account_id": account_id,
                        "amount": round(float(typical * rng.uniform(.8, 1.4)), 2),
                        "date": (candidate_time - timedelta(minutes=10 * (k + 1))).isoformat().replace("+00:00", "Z"),
                        "merchant_id": f"{account_id}-burst-{j}-{k}", "merchant_name": "New merchant", "category": unfamiliar})
            transaction = {"id": f"{account_id}-c{j}", "account_id": account_id, "amount": amount,
                "date": candidate_time.isoformat().replace("+00:00", "Z"), "merchant_id": merchant_id,
                "merchant_name": f"Regular {category.title()}" if merchant_id in normal_merchants.values() else "New merchant",
                "category": category}
            splits[split].append({"account_id": account_id, "transaction": transaction,
                                  "history": prior, "features": features(transaction, prior),
                                  "label": label, "scenario": scenario})
    return splits


def fit_student(rows: list[dict], label_indices: set[int], teacher_targets: np.ndarray | None,
                mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    chosen = np.array([i for i in range(len(rows)) if i in label_indices or
                       (teacher_targets is not None and np.isfinite(teacher_targets[i]))], dtype=int)
    x = np.stack([(feature_vector(rows[i]["features"]) - mean) / scale for i in chosen])
    y = np.array([rows[i]["label"] if i in label_indices else teacher_targets[i] for i in chosen], dtype=float)
    weights = np.array([1.0 if i in label_indices else .4 for i in chosen])
    w = np.zeros(len(FEATURE_NAMES) + 1)
    x = np.c_[np.ones(len(x)), x]
    for _ in range(400):
        logits = np.clip(x @ w, -25, 25)
        pred = 1 / (1 + np.exp(-logits))
        grad = x.T @ ((pred - y) * weights) / weights.sum()
        grad[1:] += .01 * w[1:]
        w -= .12 * grad
    return w


def train_models(train: list[dict]) -> dict:
    raw = np.stack([feature_vector(row["features"]) for row in train])
    mean = raw.mean(axis=0); scale = raw.std(axis=0); scale[scale < 1e-8] = 1
    rng = np.random.default_rng(SEED + 99)
    indices = rng.permutation(len(train))[:round(.15 * len(train))]
    labeled = set(map(int, indices))
    targets = np.array([{"LOW": .05, "REVIEW": .5, "HIGH": .95}[fixture_teacher(row["features"])["risk"]] for row in train])
    other = [i for i in range(len(train)) if i not in labeled]
    shuffled = targets.copy(); shuffled[other] = targets[rng.permutation(other)]
    models = {"independent": fit_student(train, labeled, None, mean, scale),
              "guided": fit_student(train, labeled, targets, mean, scale),
              "shuffled": fit_student(train, labeled, shuffled, mean, scale)}
    return {"models": models, "mean": mean, "scale": scale, "labeled": sorted(labeled),
            "messages": len(other), "teacher_categories": {risk: int(sum(fixture_teacher(train[i]["features"])["risk"] == risk for i in other)) for risk in RISKS}}


def score(model: np.ndarray, f: dict, mean: np.ndarray, scale: np.ndarray) -> float:
    x = (feature_vector(f) - mean) / scale
    z = float(model[0] + x @ model[1:])
    return float(1 / (1 + math.exp(-max(-25, min(25, z)))))


def risk_band(value: float) -> str:
    return "HIGH" if value >= .7 else "REVIEW" if value >= .35 else "LOW"


def evidence(f: dict) -> list[str]:
    out = []
    if f["log_amount_ratio"] > math.log1p(3): out.append("Amount is over 3× the account median")
    if f["new_merchant"]: out.append("Merchant is new to this account")
    if f["new_category"]: out.append("Spending category is new to this account")
    if f["recent_count"] >= 2: out.append(f"{f['recent_count']} purchases in the prior 24 hours")
    if f["same_merchant_count"] >= 2: out.append("Repeated purchases at this merchant")
    if f["recent_spend_ratio"] > 2.5: out.append("Recent spending is high against the account baseline")
    return out or ["No strong deviation from the available account history"]


def evaluate(models: dict, test: list[dict]) -> dict:
    results = {}
    for name, model in models["models"].items():
        predictions = []
        start = time.perf_counter()
        for row in test:
            value = score(model, row["features"], models["mean"], models["scale"])
            predictions.append({"id": row["transaction"]["id"], "account_id": row["account_id"],
                                "scenario": row["scenario"], "label": row["label"], "score": value,
                                "prediction": int(value >= .5)})
        duration_ms = 1000 * (time.perf_counter() - start) / len(test)
        tp = sum(p["label"] == 1 and p["prediction"] == 1 for p in predictions)
        fp = sum(p["label"] == 0 and p["prediction"] == 1 for p in predictions)
        fn = sum(p["label"] == 1 and p["prediction"] == 0 for p in predictions)
        tn = sum(p["label"] == 0 and p["prediction"] == 0 for p in predictions)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        results[name] = {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
            "fraud_recall": recall, "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "mean_inference_ms": duration_ms, "predictions": predictions}
    return results
