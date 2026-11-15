"""Serve the isolated Synapse Fraud dashboard and read-only local API.

Run from repository root: python -m fraud.server
"""
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

from .config import load_env
from .core import evidence, features, fixture_teacher, make_dataset, risk_band, score
from .nessie import NessieClient, NessieUnavailable
from .teacher import frontier_available, teacher_for_demo, teacher_model, teacher_provider
from .tenure_experiment import FEATURES as TENURE_FEATURES, extended_features, vector as tenure_vector

ROOT = Path(__file__).resolve().parent.parent
load_env()
ARTIFACT = json.loads((ROOT / "data" / "fraud" / "experiment.json").read_text(encoding="utf-8"))
DATASET = make_dataset()
MEAN = np.array(ARTIFACT["scaler"]["mean"])
SCALE = np.array(ARTIFACT["scaler"]["scale"])
MODELS = {name: np.array(weights) for name, weights in ARTIFACT["models"].items()}
FRONTIER_PATH = ROOT / "data" / "fraud" / "frontier-student.json"
FRONTIER_STUDENT = None
TENURE_PATH = ROOT / "data" / "fraud" / "tenure-development.json"
TENURE_STUDENTS = None
if TENURE_PATH.exists():
    candidate = json.loads(TENURE_PATH.read_text(encoding="utf-8"))
    if candidate.get("features") == list(TENURE_FEATURES):
        TENURE_STUDENTS = {"mean": np.array(candidate["mean"]),
            "scale": np.array(candidate["scale"]),
            "models": {"tenure_control": np.array(candidate["weights"]["matched_control"]),
                       "tenure_gemini": np.array(candidate["weights"]["gemini_teacher"])}}
if FRONTIER_PATH.exists():
    try:
        candidate = json.loads(FRONTIER_PATH.read_text(encoding="utf-8"))
        valid = (candidate.get("feature_names") == list(ARTIFACT["feature_names"]) and
                 len(candidate.get("weights", [])) == len(MEAN) + 1 and
                 np.allclose(candidate.get("mean", []), MEAN) and
                 np.allclose(candidate.get("scale", []), SCALE))
    except (ValueError, TypeError, AttributeError, OSError):
        valid = False
    if valid:
        FRONTIER_STUDENT = candidate
        MODELS["frontier_guided"] = np.array(candidate["weights"])
FIXTURE_IDS = sorted({row["account_id"] for row in DATASET["test"]})


def mode() -> str:
    return "nessie" if os.environ.get("NESSIE_API_KEY") else "fixture"


def source_mode(requested: str | None = None) -> str:
    selected = requested or mode()
    if selected not in {"nessie", "fixture"}:
        raise ValueError("Unknown data source")
    if selected == "nessie" and not os.environ.get("NESSIE_API_KEY"):
        raise ValueError("Nessie API key is not configured")
    return selected


def accounts(source: str | None = None) -> list[dict]:
    if source_mode(source) == "nessie":
        return NessieClient().accounts()
    return [{"id": account_id, "name": f"Mock account {account_id[-3:]}"} for account_id in FIXTURE_IDS]


def rows_for(account_id: str, source: str | None = None) -> list[dict]:
    if source_mode(source) == "fixture":
        if account_id not in FIXTURE_IDS:
            raise ValueError("Unknown fixture account")
        return [row for row in DATASET["test"] if row["account_id"] == account_id]
    purchases = NessieClient().purchases(account_id)
    rows = []
    for i, transaction in enumerate(purchases):
        history = purchases[:i]
        rows.append({"account_id": account_id, "transaction": transaction, "history": history,
                     "features": features(transaction, history), "label": None, "scenario": None})
    return rows


def scored_rows(account_id: str, source: str | None = None) -> list[dict]:
    selected_source = source_mode(source)
    scored = []
    for row in rows_for(account_id, selected_source):
        f = row["features"]
        scores = {name: round(score(model, f, MEAN, SCALE), 4) for name, model in MODELS.items()}
        extended = extended_features(row)
        if TENURE_STUDENTS:
            x = (tenure_vector(extended) - TENURE_STUDENTS["mean"]) / TENURE_STUDENTS["scale"]
            for name, weights in TENURE_STUDENTS["models"].items():
                scores[name] = round(float(1 / (1 + np.exp(-np.clip(weights[0] + x @ weights[1:], -25, 25)))), 4)
        indicators = evidence(f)
        if extended["same_merchant_count"] >= 2:
            older = extended["merchant_older_count"]
            indicators.append(f"{older} earlier purchases at this merchant more than 24 hours ago")
        scored.append({"transaction": row["transaction"], "scores": scores,
                       "bands": {name: risk_band(value) for name, value in scores.items()},
                       "evidence": indicators, "context": {"history_purchases": f["history_size"],
                         "historical_median": f["history_median"], "purchases_last_24h": f["recent_count"],
                         "same_merchant_last_24h": f["same_merchant_count"],
                         "merchant_older_count": extended["merchant_older_count"]},
                       "teacher_fixture": fixture_teacher(f),
                       "synthetic_label": row["label"] if selected_source == "fixture" else None})
    return scored


def research_summary() -> dict:
    return {"dataset": ARTIFACT["dataset"], "teacher_source": ARTIFACT["teacher_source"],
            "teacher_messages": ARTIFACT["teacher_messages"],
            "teacher_disconnected": ARTIFACT["evaluation_teacher_disconnected"],
            "success_gate_passed": ARTIFACT["success_gate_passed"],
            "arms": {name: {key: result[key] for key in
                            ("precision", "recall", "f1", "false_positive_rate", "fraud_recall", "confusion", "mean_inference_ms")}
                     for name, result in ARTIFACT["held_out"].items()}}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        # Never log a request line, path, or query supplied by a browser.
        super().log_message("request handled")

    def static_allowed(self, path: str) -> bool:
        path = unquote(path)
        if path == "/":
            return True
        parts = Path(path.lstrip("/")).parts
        if not parts or any(part.startswith(".") or part == ".." for part in parts):
            return False
        if any("\\" in part or ":" in part for part in parts):
            return False
        rel = Path(*parts)
        if rel == Path("data/fraud/frontier-student.json"):
            return False
        if rel.suffix.lower() in {".html", ".js", ".css", ".woff2", ".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".wasm"}:
            return True
        return ((rel.suffix.lower() == ".md" and parts[0] == "docs") or
                (rel.suffix.lower() == ".json" and parts[0] == "data"))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def json_response(self, value: dict, status=200):
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if not parsed.path.startswith("/api/fraud/"):
            if not self.static_allowed(parsed.path):
                return self.send_error(404, "Not found")
            return super().do_GET()
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/fraud/status":
                return self.json_response({"mode": mode(), "frontier_configured": frontier_available(),
                    "teacher_provider": teacher_provider(), "teacher_model": teacher_model(),
                    "frontier_student_trained": FRONTIER_STUDENT is not None,
                    "frontier_student_messages": FRONTIER_STUDENT["teacher_examples"] if FRONTIER_STUDENT else 0,
                    "frontier_student_provider": FRONTIER_STUDENT.get("teacher_provider", "openai") if FRONTIER_STUDENT else None,
                    "tenure_student_trained": TENURE_STUDENTS is not None,
                    "model": "local NumPy logistic classifier", "prototype": True})
            if parsed.path == "/api/fraud/accounts":
                selected_source = source_mode(query.get("source", [None])[0])
                return self.json_response({"mode": selected_source, "accounts": accounts(selected_source)})
            if parsed.path == "/api/fraud/account":
                selected_source = source_mode(query.get("source", [None])[0])
                account_id = query.get("id", [""])[0]
                if not account_id:
                    return self.json_response({"error": "Choose an account"}, 400)
                transactions = scored_rows(account_id, selected_source)
                return self.json_response({"mode": selected_source, "account_id": account_id,
                    "live_nessie": selected_source == "nessie" and bool(transactions),
                    "transactions": transactions})
            if parsed.path == "/api/fraud/research":
                return self.json_response(research_summary())
            return self.json_response({"error": "Unknown endpoint"}, 404)
        except ValueError as error:
            return self.json_response({"error": str(error)}, 400)
        except NessieUnavailable as error:
            return self.json_response({"error": str(error), "mode": "nessie"}, 502)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/fraud/frontier":
            return self.json_response({"error": "Unknown endpoint"}, 404)
        if not frontier_available():
            return self.json_response({"error": "Frontier teacher is not configured"}, 503)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2048 or length < 1:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            account_id, transaction_id = body["account_id"], body["transaction_id"]
            matches = [r for r in rows_for(account_id, body.get("source")) if r["transaction"]["id"] == transaction_id]
            if len(matches) != 1:
                raise ValueError("Transaction not found")
            result = teacher_for_demo(matches[0]["features"], use_frontier=True)
            return self.json_response(result)
        except (ValueError, KeyError, TypeError) as error:
            return self.json_response({"error": str(error)}, 400)
        except NessieUnavailable as error:
            return self.json_response({"error": str(error), "mode": "nessie"}, 502)
        except RuntimeError as error:
            return self.json_response({"error": str(error)}, 502)


if __name__ == "__main__":
    port = int(os.environ.get("SYNAPSE_FRAUD_PORT", "8767"))
    print(f"Synapse Fraud serving at http://localhost:{port}/fraud.html ({mode()} mode)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
