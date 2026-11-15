"""Optional frontier teacher. It is never used in the locked fixture-teacher study."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import load_env
from .core import FEATURE_NAMES, fixture_teacher, validate_teacher_output

CACHE_DIR = Path(__file__).resolve().parent / ".teacher-cache"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_PROMPT = ("Classify this synthetic purchase for human review using only the six supplied "
    "history features. Return LOW, REVIEW, or HIGH, a brief explanation, indicators tied to "
    "specific supplied feature values, and uncertainty. These are review priorities, not "
    "verified fraud labels. Do not invent account, merchant, or customer facts.")

SCHEMA = {"type": "object", "properties": {
    "risk": {"type": "string", "enum": ["LOW", "REVIEW", "HIGH"]},
    "explanation": {"type": "string", "description": "Brief, at most 300 characters."},
    "indicators": {"type": "array", "maxItems": 6, "items": {"type": "string", "description": "At most 70 characters, grounded in a supplied feature value."}},
    "uncertainty": {"type": "string", "description": "Brief limitation, at most 140 characters."}},
    "required": ["risk", "explanation", "indicators", "uncertainty"],
    "additionalProperties": False}


def teacher_provider() -> str:
    load_env()
    provider = os.environ.get("TEACHER_PROVIDER", "openai").lower()
    if provider not in {"openai", "gemini"}:
        raise RuntimeError("TEACHER_PROVIDER must be gemini or openai")
    return provider


def teacher_model() -> str:
    return (os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) if teacher_provider() == "gemini"
            else os.environ.get("SYNAPSE_FRAUD_TEACHER_MODEL", DEFAULT_MODEL))


def frontier_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY" if teacher_provider() == "gemini" else "OPENAI_API_KEY"))


def openai_teacher(f: dict, opener=urlopen) -> dict:
    """Send feature-only mock context, validate structured output, never log secrets."""
    load_env()
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


def gemini_teacher(f: dict, opener=urlopen, feature_names=FEATURE_NAMES, prompt=GEMINI_PROMPT) -> dict:
    """Call Gemini with feature-only mock context and a constrained JSON schema."""
    load_env()
    key = os.environ.get("GEMINI_API_KEY")
    model = teacher_model()
    if not key or not model:
        raise RuntimeError("Gemini teacher is not configured")
    limited = {name: round(float(f[name]), 3) for name in feature_names}
    payload = {"contents": [{"parts": [{"text": prompt + "\nFeatures: " + json.dumps(limited, sort_keys=True)}]}],
               "generationConfig": {"responseMimeType": "application/json",
                                    "responseJsonSchema": SCHEMA, "maxOutputTokens": 384}}
    request = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent",
                      data=json.dumps(payload).encode(),
                      headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST")
    try:
        with opener(request, timeout=30) as response:
            data = json.load(response)
        candidates = data.get("candidates") or []
        if len(candidates) != 1:
            raise RuntimeError("Gemini teacher returned no single candidate")
        reason = candidates[0].get("finishReason")
        if reason != "STOP":
            safe_reason = reason if reason in {"MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"} else "unknown"
            raise RuntimeError(f"Gemini teacher stopped ({safe_reason})")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        return validate_teacher_output(json.loads(text))
    except HTTPError as error:
        raise RuntimeError(f"Gemini teacher request failed (HTTP {error.code})") from None
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Gemini teacher request failed ({type(error).__name__})") from None
    except json.JSONDecodeError:
        raise RuntimeError("Gemini teacher returned invalid JSON") from None
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        detail = str(error) if isinstance(error, ValueError) else type(error).__name__
        raise RuntimeError(f"Gemini teacher returned invalid structured output ({detail})") from None


def frontier_teacher(f: dict, opener=urlopen) -> dict:
    return gemini_teacher(f, opener) if teacher_provider() == "gemini" else openai_teacher(f, opener)


def cached_frontier_teacher(f: dict, cache_dir: Path = CACHE_DIR, with_source: bool = False):
    """Cache only valid training responses; the key contains feature values, never IDs."""
    if not frontier_available():
        raise RuntimeError("Frontier teacher is not configured")
    provider, model = teacher_provider(), teacher_model()
    limited = {name: round(float(f[name]), 3) for name in FEATURE_NAMES}
    digest = hashlib.sha256(json.dumps([provider, model, GEMINI_PROMPT if provider == "gemini" else "openai-v1", limited], sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.exists():
        try:
            result = validate_teacher_output(json.loads(path.read_text(encoding="utf-8")))
            return (result, "cached") if with_source else result
        except (ValueError, TypeError, OSError):
            pass
    result = frontier_teacher(f)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return (result, "live") if with_source else result


def teacher_for_demo(f: dict, use_frontier: bool = False) -> dict:
    if use_frontier:
        result, freshness = cached_frontier_teacher(f, with_source=True)
        return {"source": f"{freshness} {teacher_provider().title()} teacher ({teacher_model()})", **result}
    return {"source": "deterministic rule fixture", **fixture_teacher(f)}
