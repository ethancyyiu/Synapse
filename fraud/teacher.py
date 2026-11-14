"""Optional frontier teacher. It is never used in the locked fixture-teacher study."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

from .core import FEATURE_NAMES, fixture_teacher, validate_teacher_output

CACHE_DIR = Path(__file__).resolve().parent / ".teacher-cache"
DEFAULT_MODEL = "gpt-5.6-luna"

SCHEMA = {"type": "object", "properties": {
    "risk": {"type": "string", "enum": ["LOW", "REVIEW", "HIGH"]},
    "explanation": {"type": "string"},
    "indicators": {"type": "array", "items": {"type": "string"}},
    "uncertainty": {"type": "string"}},
    "required": ["risk", "explanation", "indicators", "uncertainty"],
    "additionalProperties": False}


def frontier_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def frontier_teacher(f: dict, opener=urlopen) -> dict:
    """Send feature-only mock context, validate structured output, never log secrets."""
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("SYNAPSE_FRAUD_TEACHER_MODEL", DEFAULT_MODEL)
    if not key or not model:
        raise RuntimeError("Frontier teacher is not configured")
    limited = {name: round(float(f[name]), 3) for name in FEATURE_NAMES}
    payload = {"model": model, "store": False,
        "instructions": "Classify a synthetic purchase for human review. LOW/REVIEW/HIGH are risk categories, not verified fraud labels. Base your explanation only on supplied features. State uncertainty. No account identity or personal data is provided.",
        "input": json.dumps(limited),
        "text": {"format": {"type": "json_schema", "name": "fraud_risk", "strict": True, "schema": SCHEMA}}}
    request = Request("https://api.openai.com/v1/responses", data=json.dumps(payload).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with opener(request, timeout=25) as response:
            data = json.load(response)
        if data.get("status") != "completed":
            raise RuntimeError("Frontier teacher did not complete")
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        return validate_teacher_output(json.loads(content["text"]))
        raise RuntimeError("Frontier teacher returned no structured text")
    except Exception:
        raise RuntimeError("Frontier teacher request failed or output was invalid") from None


def cached_frontier_teacher(f: dict, cache_dir: Path = CACHE_DIR) -> dict:
    """Cache only valid training responses; the key contains feature values, never IDs."""
    if not frontier_available():
        raise RuntimeError("Frontier teacher is not configured")
    model = os.environ.get("SYNAPSE_FRAUD_TEACHER_MODEL", DEFAULT_MODEL)
    limited = {name: round(float(f[name]), 3) for name in FEATURE_NAMES}
    digest = hashlib.sha256(json.dumps([model, limited], sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.exists():
        try:
            return validate_teacher_output(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, OSError):
            pass
    result = frontier_teacher(f)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def teacher_for_demo(f: dict, use_frontier: bool = False) -> dict:
    result = frontier_teacher(f) if use_frontier else fixture_teacher(f)
    return {"source": "frontier OpenAI model" if use_frontier else "deterministic rule fixture", **result}
