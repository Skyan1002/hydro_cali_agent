"""Shared utilities for calibration agents."""

from __future__ import annotations

import base64
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..config import FROZEN_PARAMETERS, PARAM_BOUNDS
from ..parameters import ParameterSet, apply_step_guard
from ..zoned import SEP, base_name, zone_of, zoned_name
from . import llm as _llm
# ``get_client`` used to build an OpenAI client at import time here, which made
# every agent (including Claude-only arms) hard-depend on OPENAI_API_KEY. The
# lazy provider clients now live in ``llm.py``; this re-export keeps the old name.
from .llm import get_client  # noqa: F401

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.M)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

VALIDATION_MAX_RETRIES = 2
# API/transport failures get their own, longer budget: they are infrastructure,
# not a property of the model, so they must not be mistaken for a failed round.
API_MAX_RETRIES = 5
VALIDATION_COUNTER_KEYS = (
    "n_retry",
    "n_rejected",
    "n_api_retry",     # transport/provider failures that a retry recovered
    "n_api_failed",    # transport/provider failures that aborted the run
    "n_api_permanent", # of those, ones that could never succeed (context limit, credit, auth)
    "n_out_of_bound_clipped",
    "n_frozen_dropped",
    "n_unknown_param_dropped",
    # zoned runs only: how the agent addressed the partition
    "n_zone_wildcards",   # `<param>@*` entries, i.e. basin-wide moves
    "n_zone_explicit",    # entries naming one zone
)


def new_validation_counters() -> Dict[str, int]:
    return {k: 0 for k in VALIDATION_COUNTER_KEYS}


class ValidationError(ValueError):
    """Raised by ``validate_candidates`` when an LLM reply fails the schema checks.

    ``feedback`` is the message fed back to the model on retry.
    """

    def __init__(self, feedback: str):
        super().__init__(feedback)
        self.feedback = feedback


class CandidateRejected(RuntimeError):
    """An LLM stage failed validation on every attempt; the round must be skipped."""

    def __init__(self, stage: str, round_label: str, attempts: List[Dict[str, Any]]):
        self.stage = stage
        self.round_label = round_label
        self.attempts = attempts
        last = attempts[-1].get("error") if attempts else "?"
        super().__init__(f"{stage} rejected after {len(attempts)} attempt(s) at {round_label}: {last}")


class ApiUnavailable(RuntimeError):
    """The provider could not be reached at all (credit, rate limit, transport).

    Kept distinct from :class:`CandidateRejected` on purpose. A rejected candidate
    means the model produced output we could not use — that is a property of the
    model and legitimately consumes ``failure_patient``. An API outage says nothing
    about the model: counting it as a failed round silently turns an infrastructure
    problem into what looks like a poor calibration, which is exactly how the first
    Claude arm produced NSE = -0.83 when the account ran out of credit. Runs that
    hit this abort and are marked incomplete instead.
    """

    def __init__(self, stage: str, round_label: str, attempts: List[Dict[str, Any]]):
        self.stage = stage
        self.round_label = round_label
        self.attempts = attempts
        last = attempts[-1].get("error") if attempts else "?"
        super().__init__(f"{stage} could not reach the API after {len(attempts)} attempt(s) "
                         f"at {round_label}: {last}")


# --- JSON extraction --------------------------------------------------------


def extract_json_block(text: str) -> Dict[str, Any]:
    """Original greedy extraction (unchanged): first ``{`` to last ``}``."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _balanced_objects(text: str) -> List[str]:
    """Yield every top-level balanced ``{...}`` span in ``text`` (string-aware)."""
    spans: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start:i + 1])
                    start = -1
    return spans


def extract_json_robust(text: str) -> Dict[str, Any]:
    """Parse a JSON object out of ``text``; tolerant to prose and code fences.

    Order: (1) the original greedy match, so every reply the submitted code
    accepted parses identically; (2) fenced ```json blocks; (3) each balanced
    top-level object, largest first. Raises ``ValueError`` when nothing parses.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty LLM response")
    try:
        return extract_json_block(text)
    except (ValueError, json.JSONDecodeError):
        pass
    for m in _FENCE_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    for span in sorted(_balanced_objects(text), key=len, reverse=True):
        try:
            obj = json.loads(span)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("No parseable JSON object found in LLM response")


# --- schema / whitelist validation -----------------------------------------


def parameter_whitelist(display_name_map: Optional[Dict[str, str]] = None,
                        allow_wildcard: bool = True) -> set:
    """Accepted parameter names: PARAM_BOUNDS keys plus their prompt-facing labels.

    Zoned runs also accept ``<base>@*``, the all-zones shorthand advertised in the
    parameter guide, unless ``allow_wildcard`` is off -- then a wildcard fails
    validation and the model is told so, which forces one entry per zone.
    """
    names = set(PARAM_BOUNDS.keys())
    if display_name_map:
        shown = list(display_name_map.values())
        names.update(shown)
        if allow_wildcard:
            names.update(f"{base_name(s)}{SEP}*" for s in shown if SEP in s)
    return names


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    if isinstance(x, str):
        try:
            return math.isfinite(float(x))
        except ValueError:
            return False
    return False


def validate_candidates(data: Dict[str, Any],
                        key: str,
                        k: int,
                        whitelist: set,
                        *,
                        min_candidates: Optional[int] = None) -> List[Dict[str, Any]]:
    """Deterministic programmatic control on one LLM reply (R2-9).

    Checks, in order: ``key`` is a non-empty list; at least ``min_candidates``
    (default ``k``) entries; every entry is an object with an ``updates`` mapping;
    every update name is in ``whitelist``; every value is a finite number or an
    ``{"op": "*|+|-|=", "value": finite}`` dict. Returns the first ``k`` entries.
    Raises ``ValidationError`` with a model-facing message otherwise.
    """
    need = k if min_candidates is None else min_candidates
    cands = data.get(key)
    if not isinstance(cands, list) or not cands:
        raise ValidationError(
            f"The JSON object must contain a non-empty `{key}` list (got {type(cands).__name__})."
        )
    if len(cands) < need:
        raise ValidationError(
            f"`{key}` contains {len(cands)} entries but {need} were requested; return exactly {k}."
        )
    problems: List[str] = []
    for idx, cand in enumerate(cands[:k]):
        if not isinstance(cand, dict):
            problems.append(f"entry {idx} is not an object")
            continue
        updates = cand.get("updates")
        if not isinstance(updates, dict) or not updates:
            problems.append(f"entry {idx} ({cand.get('id', '?')}) has no non-empty `updates` mapping")
            continue
        for name, spec in updates.items():
            if name not in whitelist:
                problems.append(
                    f"entry {idx}: unknown parameter '{name}' (allowed: {', '.join(sorted(whitelist))})"
                )
                continue
            if isinstance(spec, dict):
                op = spec.get("op")
                if op not in ("*", "+", "-", "="):
                    problems.append(f"entry {idx}: parameter '{name}' has invalid op {op!r}")
                if not _is_finite_number(spec.get("value")):
                    problems.append(f"entry {idx}: parameter '{name}' value is not a finite number")
            elif not _is_finite_number(spec):
                problems.append(f"entry {idx}: parameter '{name}' value {spec!r} is not a finite number")
    if problems:
        shown = problems[:8]
        more = f" (+{len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
        raise ValidationError("; ".join(shown) + more)
    return list(cands[:k])


# --- validated LLM call -----------------------------------------------------


def _with_feedback(messages: List[Dict[str, Any]], feedback: str) -> List[Dict[str, Any]]:
    """Return a copy of ``messages`` with ``feedback`` appended to the last user text."""
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        content = out[i].get("content")
        if isinstance(content, str):
            out[i]["content"] = content + "\n\n" + feedback
        else:
            parts = [dict(p) for p in (content or [])]
            for j in range(len(parts) - 1, -1, -1):
                if parts[j].get("type") == "text":
                    parts[j]["text"] = parts[j].get("text", "") + "\n\n" + feedback
                    break
            else:
                parts.append({"type": "text", "text": feedback})
            out[i]["content"] = parts
        break
    return out


#: Provider errors that will never succeed on retry: the request itself is the
#: problem. Retrying these only burns the exponential backoff. The context-length
#: case is a real finding rather than an outage — INCA's history payload grows with
#: every accepted round, so a model with a smaller context window (Claude Sonnet 4.5
#: at 200k) eventually cannot be called at all, while GPT-5 never hits it.
_PERMANENT_API_PATTERNS = (
    "prompt is too long",
    "context_length_exceeded",
    "maximum context length",
    "credit balance is too low",
    "invalid_api_key",
    "authentication",
    "permission",
    "model_not_found",
    "does not exist",
)


def _is_permanent_api_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(p in msg for p in _PERMANENT_API_PATTERNS)


def _call_with_api_retries(fn: Callable[..., Any],
                           msgs: List[Dict[str, Any]],
                           *,
                           model: str,
                           provider: Optional[str],
                           thinking_budget: Optional[int],
                           stage: str,
                           round_label: str,
                           log_round: Optional[str],
                           attempt: int,
                           usage_log: List[Dict[str, Any]],
                           validation_log: List[Dict[str, Any]],
                           counters: Dict[str, int],
                           attempts: List[Dict[str, Any]],
                           max_api_retries: int) -> Any:
    """Issue one chat call, retrying transport/provider failures on their own budget.

    API failures do not consume the validation retry budget and never become a
    "failed round": if the provider stays unreachable we raise
    :class:`ApiUnavailable` so the caller aborts the run rather than recording an
    uninterpretable result.
    """
    n_images = sum(1 for m in msgs
                   for p in (m.get("content") if isinstance(m.get("content"), list) else [])
                   if isinstance(p, dict) and p.get("type") == "image_url")
    for api_try in range(max_api_retries + 1):
        try:
            return fn(msgs, model=model, provider=provider, thinking_budget=thinking_budget)
        except Exception as exc:
            usage = {"provider": provider, "model_requested": model, "model_returned": None,
                     "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
                     "reasoning_tokens": None, "cached_prompt_tokens": None, "total_tokens": None,
                     "n_images": n_images,
                     "stage": stage, "round": round_label, "log_round": log_round or round_label,
                     "attempt": attempt, "api_attempt": api_try, "used": False,
                     "error_type": "api", "error": f"{type(exc).__name__}: {exc}"[:500]}
            permanent = _is_permanent_api_error(exc)
            usage["api_error_class"] = "permanent" if permanent else "transient"
            usage_log.append(usage)
            last = permanent or api_try >= max_api_retries
            rec = {"type": "api_failed" if last else "api_retry", "stage": stage,
                   "round": log_round or round_label, "attempt": attempt, "api_attempt": api_try,
                   "error_type": "api", "api_error_class": usage["api_error_class"],
                   "error": usage["error"], "output_text": None}
            attempts.append(rec)
            validation_log.append(rec)
            if permanent:
                counters["n_api_permanent"] = counters.get("n_api_permanent", 0) + 1
            if not last:
                counters["n_api_retry"] = counters.get("n_api_retry", 0) + 1
                time.sleep(min(60.0, 5.0 * (2 ** api_try)))
                continue
            counters["n_api_failed"] = counters.get("n_api_failed", 0) + 1
            raise ApiUnavailable(stage, log_round or round_label, attempts) from exc
    raise ApiUnavailable(stage, log_round or round_label, attempts)


def call_with_validation(messages: List[Dict[str, Any]],
                         *,
                         model: str,
                         provider: Optional[str],
                         stage: str,
                         round_label: str,
                         parse: Callable[[Any], Any],
                         usage_log: List[Dict[str, Any]],
                         validation_log: List[Dict[str, Any]],
                         counters: Dict[str, int],
                         thinking_budget: Optional[int] = None,
                         expect_json: bool = True,
                         format_hint: str = "Return ONLY the JSON object.",
                         max_retries: int = VALIDATION_MAX_RETRIES,
                         max_api_retries: int = API_MAX_RETRIES,
                         log_round: Optional[str] = None,
                         chat_fn: Optional[Callable[..., Any]] = None) -> Tuple[Any, Any, Any, Dict[str, Any], List[Dict[str, Any]]]:
    """One LLM call wrapped in the deterministic programmatic control layer (R2-9).

    Flow: call -> extract JSON (when ``expect_json``) -> ``parse`` (schema /
    whitelist / finiteness checks, raising ``ValidationError``/``ValueError``).
    A failed attempt feeds the concrete error back into the end of the user
    prompt ("Your previous reply failed validation: ... <format_hint>") and the
    call is retried at most ``max_retries`` times. Provider/API exceptions are
    retried within the same budget without feedback text. Every attempt is
    appended to ``usage_log`` (``used`` is False for failed attempts) and every
    failure to ``validation_log``; ``counters['n_retry']`` / ``['n_rejected']``
    are updated. When every attempt fails ``CandidateRejected`` is raised so the
    caller can treat the round as a failure round instead of dying.

    Returns ``(response, data, result, usage, attempts)`` of the successful call.
    ``chat_fn`` defaults to ``hydrocalib.agents.llm.chat`` looked up at call time
    (so it can be monkeypatched in tests).
    """
    attempts: List[Dict[str, Any]] = []
    msgs = [dict(m) for m in messages]
    feedback: Optional[str] = None
    for attempt in range(max_retries + 1):
        if feedback:
            msgs = _with_feedback(messages, feedback)
        fn = chat_fn or _llm.chat
        usage: Dict[str, Any]
        try:
            response = _call_with_api_retries(
                fn, msgs, model=model, provider=provider, thinking_budget=thinking_budget,
                stage=stage, round_label=round_label, log_round=log_round, attempt=attempt,
                usage_log=usage_log, validation_log=validation_log, counters=counters,
                attempts=attempts, max_api_retries=max_api_retries)
        except ApiUnavailable:
            raise
        except Exception as exc:  # unreachable: kept so the original path still compiles
            usage = {"provider": provider, "model_requested": model, "model_returned": None,
                     "latency_s": None, "prompt_tokens": None, "completion_tokens": None,
                     "reasoning_tokens": None, "cached_prompt_tokens": None, "total_tokens": None,
                     "n_images": sum(1 for m in msgs for p in (m.get("content") if isinstance(m.get("content"), list) else [])
                                     if isinstance(p, dict) and p.get("type") == "image_url")}
            usage.update({"stage": stage, "round": round_label, "log_round": log_round or round_label,
                          "attempt": attempt, "used": False,
                          "error_type": "api", "error": f"{type(exc).__name__}: {exc}"[:500]})
            usage_log.append(usage)
            rec = {"type": "retry" if attempt < max_retries else "reject", "stage": stage,
                   "round": log_round or round_label, "attempt": attempt, "error_type": "api",
                   "error": usage["error"], "output_text": None}
            attempts.append(rec)
            validation_log.append(rec)
            if attempt < max_retries:
                counters["n_retry"] = counters.get("n_retry", 0) + 1
                time.sleep(min(30.0, 5.0 * (attempt + 1)))
                # no feedback text: the model never saw the prompt
                continue
            break

        usage = response.usage_record()
        usage.update({"stage": stage, "round": round_label, "log_round": log_round or round_label,
                      "attempt": attempt, "used": True})
        usage_log.append(usage)
        try:
            if expect_json:
                data = extract_json_robust(response.text)
                result = parse(data)
            else:
                data = None
                result = parse(response.text)
        except (ValidationError, ValueError, TypeError, KeyError, AttributeError) as exc:
            err = getattr(exc, "feedback", None) or f"{type(exc).__name__}: {exc}"
            usage["used"] = False
            usage["error_type"] = "validation"
            usage["error"] = str(err)[:500]
            rec = {"type": "retry" if attempt < max_retries else "reject", "stage": stage,
                   "round": log_round or round_label, "attempt": attempt, "error_type": "validation",
                   "error": str(err)[:1000], "stop_reason": usage.get("stop_reason"),
                   "output_text": (response.text or "")[:4000]}
            attempts.append(rec)
            validation_log.append(rec)
            if attempt < max_retries:
                counters["n_retry"] = counters.get("n_retry", 0) + 1
                feedback = f"Your previous reply failed validation: {err} {format_hint}"
                continue
            break
        return response, data, result, usage, attempts

    counters["n_rejected"] = counters.get("n_rejected", 0) + 1
    raise CandidateRejected(stage, log_round or round_label, attempts)


def redact_history_block(prompt_text: str) -> str:
    """Remove bulky history payloads from a JSON prompt before logging."""

    try:
        payload = json.loads(prompt_text)
    except Exception:
        return prompt_text

    if "history_payload" in payload:
        payload["history_payload"] = "[omitted]"
    return json.dumps(payload, ensure_ascii=False, indent=2)


# --- applying updates (behaviour unchanged, now instrumented) ---------------


def _expand_zone_wildcards(updates: Dict[str, Any],
                           known: Iterable[str],
                           stats: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Turn ``wm@*`` into one entry per zone present in the parameter set.

    An explicit zone always wins over the wildcard, whatever order they arrive in,
    so ``{"wm@*": 1.5, "wm@V3": 0.8}`` means "1.5 everywhere except V3". Counts how
    many entries arrived as wildcards versus named a zone, which is the statistic that
    says whether the agent moved the basin as a whole or zone by zone. Both counts are
    taken before the early return, or a candidate that named only explicit zones would
    contribute nothing and the ratio would read as more global than it is.
    """
    wildcards = {base_name(k): v for k, v in updates.items() if zone_of(k) == "*"}
    if stats is not None:
        stats["n_zone_wildcards"] = stats.get("n_zone_wildcards", 0) + len(wildcards)
        stats["n_zone_explicit"] = stats.get("n_zone_explicit", 0) + sum(
            1 for k in updates if zone_of(k) not in (None, "*"))
    if not wildcards:
        return updates
    zones = []
    for name in known:
        z = zone_of(name)
        if z is not None and z not in zones:
            zones.append(z)
    out: Dict[str, Any] = {}
    for base, spec in wildcards.items():
        for z in zones:
            out[zoned_name(base, z)] = spec
    for key, spec in updates.items():
        if zone_of(key) != "*":
            out[key] = spec
    return out


def coerce_updates(params: ParameterSet,
                   updates: Dict[str, Any],
                   stats: Optional[Dict[str, int]] = None) -> ParameterSet:
    """Apply LLM updates to ``params`` exactly as before.

    Unknown / frozen names are still dropped and out-of-bound values still
    clipped; the only addition is that these events are counted into ``stats``
    (keys ``n_unknown_param_dropped``, ``n_frozen_dropped``,
    ``n_out_of_bound_clipped``) when a dict is supplied.
    """
    new_vals = params.values.copy()
    for name, spec in _expand_zone_wildcards(updates, new_vals.keys(), stats).items():
        if name not in new_vals:
            if stats is not None:
                stats["n_unknown_param_dropped"] = stats.get("n_unknown_param_dropped", 0) + 1
            continue
        if base_name(name) in FROZEN_PARAMETERS:
            if stats is not None:
                stats["n_frozen_dropped"] = stats.get("n_frozen_dropped", 0) + 1
            continue
        old = new_vals[name]
        if isinstance(spec, dict) and "op" in spec and "value" in spec:
            op = spec["op"]
            val = float(spec["value"])
            if op == "*":
                candidate = old * val
            elif op == "+":
                candidate = old + val
            elif op == "-":
                candidate = old - val
            elif op == "=":
                candidate = val
            else:
                candidate = old
        else:
            candidate = float(spec)
        guarded = apply_step_guard(name, old, candidate)
        if stats is not None and guarded != candidate:
            stats["n_out_of_bound_clipped"] = stats.get("n_out_of_bound_clipped", 0) + 1
        new_vals[name] = guarded
    return ParameterSet(new_vals)


def b64_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


__all__ = [
    "get_client",
    "ValidationError",
    "extract_json_block",
    "extract_json_robust",
    "parameter_whitelist",
    "validate_candidates",
    "coerce_updates",
    "b64_image",
    "redact_history_block",
    "call_with_validation",
    "CandidateRejected",
    "new_validation_counters",
    "VALIDATION_MAX_RETRIES",
]
