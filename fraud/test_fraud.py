import unittest
from unittest.mock import patch
import io
import json
import os
from pathlib import Path

import numpy as np

from fraud.core import (FEATURE_NAMES, evaluate, features, fixture_teacher, make_dataset,
                        normalize_purchase, score, train_models, validate_teacher_output)
from fraud.config import load_env
from fraud.nessie import NessieClient, NessieUnavailable
from fraud.server import Handler, mode
from fraud.teacher import SCHEMA, cached_frontier_teacher, frontier_available, frontier_teacher
from fraud import train_frontier
from tools.serve import NoCache


class FraudChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_dataset()
        cls.models = train_models(cls.data["train"])

    def test_account_disjoint_and_hard_negatives(self):
        ids = [{r["account_id"] for r in self.data[name]} for name in ("train", "development", "test")]
        self.assertTrue(ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]))
        self.assertEqual([len(self.data[name]) for name in ("train", "development", "test")], [480, 120, 120])
        for scenario in ("large_known_benign", "new_small_benign", "known_burst_benign"):
            self.assertTrue(all(r["label"] == 0 for r in self.data["test"] if r["scenario"] == scenario))

    def test_student_updates_and_teacher_disconnection(self):
        self.assertEqual(self.models["messages"], 408)
        self.assertTrue(all(np.linalg.norm(w) > 0 for w in self.models["models"].values()))
        with patch("fraud.core.fixture_teacher", side_effect=AssertionError("teacher called at evaluation")):
            result = evaluate(self.models, self.data["development"])
        self.assertEqual(len(result["guided"]["predictions"]), 120)
        saved = json.loads((Path(__file__).resolve().parent.parent / "data/fraud/experiment.json").read_text(encoding="utf-8"))
        for name, weights in self.models["models"].items():
            np.testing.assert_allclose(weights, saved["models"][name])

    def test_history_and_normalization(self):
        txn = normalize_purchase({"_id": "p1", "merchant_id": "m1", "amount": 90,
                                  "purchase_date": "2026-01-03", "category": "fuel"},
                                 {"m1": {"name": "Mock Fuel", "category": "fuel"}}, "a1")
        self.assertEqual(txn["merchant_name"], "Mock Fuel")
        prior = [{**txn, "id": "p0", "amount": 10, "date": "2026-01-02T00:00:00Z"}]
        f = features(txn, prior)
        self.assertEqual(f["new_merchant"], 0)
        self.assertGreater(f["log_amount_ratio"], 2)

    def test_teacher_schema(self):
        f = self.data["train"][0]["features"]
        self.assertIn(fixture_teacher(f)["risk"], ("LOW", "REVIEW", "HIGH"))
        with self.assertRaises(ValueError): validate_teacher_output({"risk": "FRAUD", "explanation": "x", "indicators": []})
        self.assertEqual(SCHEMA["properties"]["risk"]["enum"], ["LOW", "REVIEW", "HIGH"])

    def test_nessie_key_server_only_and_error_redaction(self):
        def fake_open(request, timeout):
            raise RuntimeError("unused")
        with self.assertRaises(NessieUnavailable): NessieClient(key="")
        client = NessieClient(key="secret123", base_url="http://127.0.0.1:9999", opener=fake_open)
        with self.assertRaises(NessieUnavailable) as error: client.accounts()
        self.assertNotIn("secret123", str(error.exception))

    def test_nessie_read_only_normalization(self):
        seen = []
        def fake_open(request, timeout):
            seen.append((request.get_method(), request.full_url))
            if "/accounts/a1/purchases" in request.full_url:
                value = [{"_id": "p1", "amount": 18, "purchase_date": "2026-01-04", "merchant_id": "m1"}]
            elif "/merchants" in request.full_url:
                value = [{"_id": "m1", "name": "Mock Store", "category": "groceries"}]
            else:
                value = [{"_id": "a1", "type": "Checking"}]
            return io.BytesIO(json.dumps(value).encode())
        client = NessieClient(key="test-secret", opener=fake_open)
        self.assertEqual(client.accounts()[0]["id"], "a1")
        purchase = client.purchases("a1")[0]
        self.assertEqual(purchase["merchant_name"], "Mock Store")
        self.assertEqual(purchase["category"], "groceries")
        self.assertTrue(all(method == "GET" for method, _ in seen))
        self.assertTrue(all("key=test-secret" in url for _, url in seen))

    def test_nessie_v2_bearer_and_purchase_cents(self):
        seen = []
        def fake_open(request, timeout):
            seen.append(request)
            if "/transactions" in request.full_url:
                value = {"items": [{"id": "p2", "kind": "purchase", "amount_cents": 1825,
                                     "created_at": "2026-01-04T12:00:00Z", "merchant_id": "m2"},
                                    {"id": "d2", "kind": "deposit", "amount_cents": 5000,
                                     "created_at": "2026-01-04T13:00:00Z"}]}
            elif "/merchants" in request.full_url:
                value = {"items": [{"id": "m2", "name": "Mock Cafe", "category": "dining"}]}
            else:
                value = {"items": [{"id": "a2", "type": "Checking", "nickname": "Everyday"}]}
            return io.BytesIO(json.dumps(value).encode())
        client = NessieClient(key="nsk_test-secret", opener=fake_open)
        self.assertEqual(client.accounts()[0]["name"], "Everyday · a2")
        purchases = client.purchases("a2")
        self.assertEqual(len(purchases), 1)
        self.assertEqual(purchases[0]["amount"], 18.25)
        self.assertEqual(purchases[0]["merchant_name"], "Mock Cafe")
        self.assertTrue(all(request.get_method() == "GET" for request in seen))
        self.assertTrue(all("key=" not in request.full_url for request in seen))
        self.assertTrue(all(request.get_header("Authorization") == "Bearer nsk_test-secret" for request in seen))

    def test_frontier_schema_and_minimized_request(self):
        sent = []
        def fake_open(request, timeout):
            sent.append(json.loads(request.data))
            output = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text",
                "text": json.dumps({"risk": "REVIEW", "explanation": "Review the burst.",
                                    "indicators": ["Recent activity"], "uncertainty": "Limited history"})}]}]}
            return io.BytesIO(json.dumps(output).encode())
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret", "SYNAPSE_FRAUD_TEACHER_MODEL": "test-model"}):
            result = frontier_teacher(self.data["train"][0]["features"], opener=fake_open)
        self.assertEqual(result["risk"], "REVIEW")
        self.assertEqual(sent[0]["text"]["format"]["type"], "json_schema")
        self.assertNotIn("account_id", sent[0]["input"])
        self.assertNotIn("merchant_id", sent[0]["input"])

    def test_optional_frontier_training_and_cache_without_live_api(self):
        f = self.data["train"][0]["features"]
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-secret"}):
            with patch("fraud.teacher.frontier_teacher", return_value=fixture_teacher(f)) as fake_teacher, \
                 patch.object(Path, "exists", return_value=False), patch.object(Path, "mkdir"), \
                 patch.object(Path, "write_text") as write, patch.object(Path, "replace"):
                self.assertEqual(cached_frontier_teacher(f, Path("cache"))["risk"], fixture_teacher(f)["risk"])
                fake_teacher.assert_called_once()
                write.assert_called_once()
            with patch("fraud.teacher.frontier_teacher") as fake_teacher, \
                 patch.object(Path, "exists", return_value=True), \
                 patch.object(Path, "read_text", return_value=json.dumps(fixture_teacher(f))):
                self.assertEqual(cached_frontier_teacher(f, Path("cache"))["risk"], fixture_teacher(f)["risk"])
                fake_teacher.assert_not_called()
            with patch.object(train_frontier, "cached_frontier_teacher", return_value=fixture_teacher(f)) as teacher:
                result = train_frontier.train(2, save=False)
            self.assertEqual(teacher.call_count, 2)
            self.assertEqual(result["teacher_examples"], 2)
            self.assertEqual(result["true_label_examples"], 72)
            self.assertEqual(len(result["weights"]), 7)

    def test_static_server_keeps_private_files_out(self):
        handler = Handler.__new__(Handler)
        for path in ("/fraud.html", "/fraud/app.js", "/docs/synapse-fraud-research.md",
                     "/data/fraud/experiment.json", "/index.html"):
            self.assertTrue(handler.static_allowed(path), path)
        for path in ("/.git/config", "/%2egit/config", "/fraud/core.py", "/package.json",
                     "/docs/../.git/config", "/.env", "/requirements.txt",
                     "/data/fraud/frontier-student.json", "/fraud/.teacher-cache/x.json"):
            self.assertFalse(handler.static_allowed(path), path)
        old_handler = NoCache.__new__(NoCache)
        for path in ("/.env", "/%2eenv", "/.git/config", "/fraud/.teacher-cache/x.json", "/fraud/config.py"):
            old_handler.path = path
            self.assertTrue(old_handler.private_path(), path)
        old_handler.path = "/index.html"
        self.assertFalse(old_handler.private_path())

    def test_env_loader_keeps_process_precedence(self):
        contents = "NESSIE_API_KEY=from-file\nOPENAI_API_KEY='quoted-key'\nSYNAPSE_FRAUD_TEACHER_MODEL=gpt-5.6-luna\nBAD_VARIABLE=ignore\n"
        with patch.dict("os.environ", {"NESSIE_API_KEY": "from-process"}, clear=True), \
             patch.object(Path, "read_text", return_value=contents):
            load_env(Path("unused.env"))
            self.assertEqual(os.environ["NESSIE_API_KEY"], "from-process")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "quoted-key")
            self.assertNotIn("BAD_VARIABLE", os.environ)
            self.assertEqual(mode(), "nessie")
            self.assertTrue(frontier_available())
            self.assertIsInstance(NessieClient(), NessieClient)


if __name__ == "__main__": unittest.main()
