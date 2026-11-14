"""Load local Synapse Fraud settings without adding a dotenv dependency."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {"NESSIE_API_KEY", "OPENAI_API_KEY", "SYNAPSE_FRAUD_TEACHER_MODEL",
           "NESSIE_BASE_URL", "SYNAPSE_FRAUD_PORT"}


def load_env(path: Path | None = None) -> None:
    """Read simple KEY=VALUE lines; existing process environment takes precedence."""
    source = path or ROOT / ".env"
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if name not in ALLOWED:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(name, value)
