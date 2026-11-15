"""One approved, resumable set of fictional Nessie v2 sandbox records.

Run only after explicit approval of the exact records below. This module is not
imported by the read-only dashboard server.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import load_env
from .nessie import V2_BASE_URL

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "fraud" / "nessie-demo-records.json"
CUSTOMER = {"first_name": "Alex", "last_name": "Morgan", "address": {
    "street_number": "1500", "street_name": "Main St", "city": "McLean", "state": "VA", "zip": "22101"}}
ACCOUNT = {"type": "Checking", "nickname": "Synapse Demo", "balance_cents": 100000}
MERCHANTS = [{"name": "Corner Market", "category": "groceries"},
             {"name": "Nova Electronics", "category": "electronics"}]
PURCHASES = [("Corner Market", 1842), ("Corner Market", 2115),
             ("Corner Market", 1988), ("Nova Electronics", 17000),
             ("Nova Electronics", 18500), ("Nova Electronics", 19900)]


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE)


def post(path: str, body: dict, key: str) -> dict:
    request = Request(V2_BASE_URL + path, data=json.dumps(body).encode(), method="POST",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                               "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            data = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Nessie sandbox write failed (HTTP {error.code})") from None
    except (URLError, TimeoutError, ValueError) as error:
        raise RuntimeError(f"Nessie sandbox write failed ({type(error).__name__})") from None
    if not isinstance(data, dict) or not data.get("id"):
        raise RuntimeError("Nessie sandbox write returned no resource ID; inspect remote state before retrying")
    return data


def main() -> None:
    load_env()
    key = os.environ.get("NESSIE_API_KEY", "")
    if not key.startswith("nsk_"):
        raise RuntimeError("An nsk_ Nessie v2 sandbox key is required")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
        "customer_id": None, "account_id": None, "merchants": {}, "purchase_ids": []}
    if not state["customer_id"]:
        state["customer_id"] = post("/customers", CUSTOMER, key)["id"]
        save(state)
    if not state["account_id"]:
        path = "/customers/" + quote(state["customer_id"], safe="") + "/accounts"
        state["account_id"] = post(path, ACCOUNT, key)["id"]
        save(state)
    for merchant in MERCHANTS:
        if merchant["name"] not in state["merchants"]:
            state["merchants"][merchant["name"]] = post("/merchants", merchant, key)["id"]
            save(state)
    for index in range(len(state["purchase_ids"]), len(PURCHASES)):
        name, cents = PURCHASES[index]
        path = "/accounts/" + quote(state["account_id"], safe="") + "/purchases"
        result = post(path, {"merchant_id": state["merchants"][name], "amount_cents": cents}, key)
        state["purchase_ids"].append(result["id"])
        save(state)
    print(f"Nessie sandbox records ready: 1 customer, 1 account, 2 merchants, {len(state['purchase_ids'])} purchases")


if __name__ == "__main__":
    main()
