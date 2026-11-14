"""Read-only server-side Nessie adapter. Credentials never leave this module."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .core import normalize_purchase

BASE_URL = "https://api.nessieisreal.com"


class NessieUnavailable(RuntimeError):
    pass


class NessieClient:
    def __init__(self, key: str | None = None, base_url: str | None = None, opener=urlopen):
        self._key = key if key is not None else os.environ.get("NESSIE_API_KEY", "")
        self._base = (base_url or os.environ.get("NESSIE_BASE_URL") or BASE_URL).rstrip("/")
        self._opener = opener
        if not self._key:
            raise NessieUnavailable("Nessie credentials are not configured")
        if not self._base.startswith("https://") and not self._base.startswith("http://127.0.0.1"):
            raise NessieUnavailable("Nessie base URL must use HTTPS")

    def _get(self, path: str):
        # Never include the request URL, which contains the API key, in errors or logs.
        url = f"{self._base}{path}?{urlencode({'key': self._key})}"
        request = Request(url, headers={"Accept": "application/json"}, method="GET")
        try:
            with self._opener(request, timeout=10) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ValueError, RuntimeError) as error:
            raise NessieUnavailable(f"Nessie read failed ({type(error).__name__})") from None

    @staticmethod
    def _rows(value, key: str):
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
        raise NessieUnavailable("Unexpected Nessie response shape")

    def accounts(self) -> list[dict]:
        rows = self._rows(self._get("/accounts"), "accounts")
        return [{"id": str(a.get("_id") or a.get("id")),
                 "name": f"Mock {a.get('type', 'account')} · {str(a.get('_id') or a.get('id'))[-6:]}"}
                for a in rows if a.get("_id") or a.get("id")]

    def purchases(self, account_id: str) -> list[dict]:
        if account_id not in {a["id"] for a in self.accounts()}:
            raise NessieUnavailable("Account not found in accessible mock accounts")
        merchants = self._rows(self._get("/merchants"), "merchants")
        merchant_map = {str(m.get("_id") or m.get("id")): m for m in merchants}
        rows = self._rows(self._get(f"/accounts/{quote(account_id, safe='')}/purchases"), "purchases")
        normalized = []
        for row in rows:
            try:
                normalized.append(normalize_purchase(row, merchant_map, account_id))
            except (ValueError, TypeError, AttributeError):
                continue
        return sorted(normalized, key=lambda p: (p["date"], p["id"]))
