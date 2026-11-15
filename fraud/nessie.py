"""Read-only server-side Nessie adapter. Credentials never leave this module."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .config import load_env
from .core import normalize_purchase

V1_BASE_URL = "https://api.nessieisreal.com"
V2_BASE_URL = "https://v2-prod-api.nessieisreal.com"


class NessieUnavailable(RuntimeError):
    pass


class NessieClient:
    def __init__(self, key: str | None = None, base_url: str | None = None, opener=urlopen):
        load_env()
        self._key = key if key is not None else os.environ.get("NESSIE_API_KEY", "")
        self._version = 2 if self._key.startswith("nsk_") else 1
        default_base = V2_BASE_URL if self._version == 2 else V1_BASE_URL
        self._base = (base_url or os.environ.get("NESSIE_BASE_URL") or default_base).rstrip("/")
        self._opener = opener
        if not self._key:
            raise NessieUnavailable("Nessie credentials are not configured")
        if not self._base.startswith("https://") and not self._base.startswith("http://127.0.0.1"):
            raise NessieUnavailable("Nessie base URL must use HTTPS")

    def _get(self, path: str, params: dict | None = None):
        query = dict(params or {})
        headers = {"Accept": "application/json"}
        if self._version == 2:
            headers["Authorization"] = f"Bearer {self._key}"
        else:
            query["key"] = self._key
        url = f"{self._base}{path}" + (f"?{urlencode(query)}" if query else "")
        request = Request(url, headers=headers, method="GET")
        try:
            with self._opener(request, timeout=10) as response:
                return json.load(response)
        except HTTPError as error:
            raise NessieUnavailable(f"Nessie read failed (HTTP {error.code})") from None
        except (URLError, TimeoutError, ValueError, RuntimeError) as error:
            raise NessieUnavailable(f"Nessie read failed ({type(error).__name__})") from None

    @staticmethod
    def _rows(value, key: str):
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and value.get("items") is None and "items" in value:
            return []
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            return value["items"]
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
        raise NessieUnavailable("Unexpected Nessie response shape")

    def _list(self, path: str, key: str) -> list[dict]:
        if self._version == 1:
            return self._rows(self._get(path), key)
        rows, cursor = [], None
        for _ in range(10):
            page = self._get(path, {"limit": 100, **({"cursor": cursor} if cursor else {})})
            rows.extend(self._rows(page, key))
            cursor = page.get("next_cursor")
            if not cursor:
                return rows
        raise NessieUnavailable("Nessie list exceeded the supported page limit")

    def accounts(self) -> list[dict]:
        rows = self._list("/accounts", "accounts")
        return [{"id": str(a.get("_id") or a.get("id")),
                 "name": f"{a.get('nickname') or 'Mock ' + a.get('type', 'account')} · {str(a.get('_id') or a.get('id'))[-6:]}"}
                for a in rows if a.get("_id") or a.get("id")]

    def purchases(self, account_id: str) -> list[dict]:
        if account_id not in {a["id"] for a in self.accounts()}:
            raise NessieUnavailable("Account not found in accessible mock accounts")
        merchants = self._list("/merchants", "merchants")
        merchant_map = {str(m.get("_id") or m.get("id")): m for m in merchants}
        if self._version == 2:
            rows = self._rows(self._get(f"/accounts/{quote(account_id, safe='')}/transactions", {"limit": 500}), "transactions")
            rows = [row for row in rows if row.get("kind") == "purchase"]
        else:
            rows = self._rows(self._get(f"/accounts/{quote(account_id, safe='')}/purchases"), "purchases")
        normalized = []
        for row in rows:
            try:
                normalized.append(normalize_purchase(row, merchant_map, account_id))
            except (ValueError, TypeError, AttributeError):
                continue
        return sorted(normalized, key=lambda p: (p["date"], p["id"]))
