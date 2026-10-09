"""Provider-agnostic chat layer with full usage accounting.

Why this exists
---------------
The submitted version called ``client.chat.completions.create()`` directly in
three places and logged only ``output_text``. Two consequences:

1. **Cost was not recoverable.** GPT-5 is a reasoning model: in a measured
   image-diagnosis call, 448 of 496 completion tokens were *reasoning* tokens,
   which are billed but never appear in ``output_text``. Estimating cost from
   the logged text under-counts output by roughly an order of magnitude.
2. **The engine could not be swapped**, so "is this a GPT-5 capability or a
   framework contribution?" could not be tested.

This module fixes both: one ``chat()`` entry point across OpenAI / Anthropic /
Google, returning text *plus* a normalized usage record (including reasoning
tokens), the resolved model snapshot, and wall-clock latency.

Messages use OpenAI shape; provider-specific translation happens here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

load_dotenv()

# --- provider inference -----------------------------------------------------

_PROVIDER_PREFIXES = {
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt"),
    "anthropic": ("claude-",),
    "google": ("gemini-",),
}


def infer_provider(model: str) -> str:
    low = model.lower()
    for provider, prefixes in _PROVIDER_PREFIXES.items():
        if low.startswith(prefixes):
            return provider
    raise ValueError(f"Cannot infer provider for model {model!r}; pass provider explicitly.")


# --- normalized response ----------------------------------------------------


@dataclass
class LLMResponse:
    """One chat completion plus everything needed for a cost ledger."""

    text: str
    provider: str
    model_requested: str
    model_returned: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    n_images: int = 0
    raw_usage: Dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_mode: Optional[str] = None  # None | "enabled" | "adaptive"

    def usage_record(self) -> Dict[str, Any]:
        """Compact dict to embed in ``detail_logs`` entries."""
        return {
            "provider": self.provider,
            "model_requested": self.model_requested,
            "model_returned": self.model_returned,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "total_tokens": self.total_tokens,
            "n_images": self.n_images,
            "stop_reason": self.stop_reason,
            "thinking_budget": self.thinking_budget,
            "thinking_mode": self.thinking_mode,
        }


# --- lazy clients -----------------------------------------------------------

_clients: Dict[str, Any] = {}


def _client(provider: str) -> Any:
    if provider in _clients:
        return _clients[provider]
    if provider == "openai":
        from openai import OpenAI

        c = OpenAI()
    elif provider == "anthropic":
        import anthropic

        c = anthropic.Anthropic()
    elif provider == "google":
        from google import genai

        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        c = genai.Client(api_key=key)
    else:
        raise ValueError(f"Unknown provider {provider!r}")
    _clients[provider] = c
    return c


def get_client():
    """Backwards-compatible accessor for the OpenAI client."""
    return _client("openai")


# --- message translation ----------------------------------------------------


def _split_system(messages: Sequence[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    system_parts, rest = [], []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            system_parts.append(c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if isinstance(p, dict)))
        else:
            rest.append(m)
    return "\n".join(system_parts), rest


def _parts(content: Any) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _b64_from_data_url(url: str) -> tuple[str, str]:
    """'data:image/png;base64,XXXX' -> ('image/png', 'XXXX')"""
    header, _, payload = url.partition(",")
    media = header.split(";")[0].removeprefix("data:") or "image/png"
    return media, payload


def _count_images(messages: Sequence[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        for p in _parts(m.get("content", "")):
            if p.get("type") == "image_url":
                n += 1
    return n


# --- the one entry point ----------------------------------------------------


_THINKING_WARNED: set = set()


def chat(messages: Sequence[Dict[str, Any]],
         model: str,
         provider: Optional[str] = None,
         *,
         max_tokens: int = 16000,
         temperature: Optional[float] = None,
         timeout: Optional[float] = None,
         thinking_budget: Optional[int] = None) -> LLMResponse:
    """Run one chat completion and return text + normalized usage.

    ``messages`` follow the OpenAI schema. ``temperature`` is passed through only
    when not None and only to providers that accept it (GPT-5 reasoning models
    reject it), so the default keeps provider defaults rather than silently
    claiming a setting that was never applied.

    ``thinking_budget`` (Anthropic only) enables extended thinking with the given
    ``budget_tokens``. ``None`` (default) leaves the request exactly as before, so
    every existing arm is unaffected. When set: Anthropic requires
    ``max_tokens > budget_tokens`` and rejects ``temperature`` together with
    thinking, so ``max_tokens`` is raised when needed and ``temperature`` is
    omitted. Only ``text`` blocks are returned as ``text``; the thinking token
    count is recorded as ``reasoning_tokens`` when the usage object carries it.
    Model generations that no longer accept ``budget_tokens`` (Claude 4.6+ return
    HTTP 400 for it) are retried once with ``{"type": "adaptive"}`` and the
    resolved mode is recorded in ``thinking_mode``.
    """
    provider = provider or infer_provider(model)
    n_images = _count_images(messages)
    t0 = time.perf_counter()

    if thinking_budget is not None and provider != "anthropic" and provider not in _THINKING_WARNED:
        _THINKING_WARNED.add(provider)
        print(f"[Warn] thinking_budget={thinking_budget} is only applied to the anthropic "
              f"provider; ignored for provider={provider!r}.")

    if provider == "openai":
        kwargs: Dict[str, Any] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout
        r = _client(provider).chat.completions.create(**kwargs)
        text = r.choices[0].message.content or ""
        u = r.usage
        details = getattr(u, "completion_tokens_details", None)
        prompt_details = getattr(u, "prompt_tokens_details", None)
        raw_usage = u.model_dump() if hasattr(u, "model_dump") else {}
        # Backend configuration fingerprint (finding 20): the only signal that the
        # served configuration changed between two runs of the same snapshot.
        raw_usage["system_fingerprint"] = getattr(r, "system_fingerprint", None)
        raw_usage["service_tier"] = getattr(r, "service_tier", None)
        resp = LLMResponse(
            text=text, provider=provider, model_requested=model,
            model_returned=getattr(r, "model", model),
            latency_s=time.perf_counter() - t0,
            prompt_tokens=getattr(u, "prompt_tokens", None),
            completion_tokens=getattr(u, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            cached_prompt_tokens=getattr(prompt_details, "cached_tokens", None) if prompt_details else None,
            total_tokens=getattr(u, "total_tokens", None),
            n_images=n_images,
            raw_usage=raw_usage,
            stop_reason=getattr(r.choices[0], "finish_reason", None),
        )
        return resp

    if provider == "anthropic":
        system, rest = _split_system(messages)
        conv = []
        for m in rest:
            blocks = []
            for p in _parts(m.get("content", "")):
                if p.get("type") == "text":
                    blocks.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    media, data = _b64_from_data_url(p["image_url"]["url"])
                    blocks.append({"type": "image",
                                   "source": {"type": "base64", "media_type": media, "data": data}})
            conv.append({"role": m.get("role", "user"), "content": blocks})
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": conv}
        if system:
            kwargs["system"] = system
        thinking_mode: Optional[str] = None
        if thinking_budget is not None:
            budget = int(thinking_budget)
            if budget < 1024:
                raise ValueError("thinking_budget must be >= 1024 tokens for Anthropic models")
            # Anthropic: max_tokens must exceed budget_tokens; temperature is not
            # accepted together with thinking, so it is deliberately not passed.
            if kwargs["max_tokens"] <= budget:
                kwargs["max_tokens"] = budget + max_tokens
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            thinking_mode = "enabled"
        elif temperature is not None:
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout
        client = _client(provider)
        try:
            r = client.messages.create(**kwargs)
        except Exception as exc:  # budget_tokens rejected by newer model generations
            msg = str(exc)
            if thinking_mode == "enabled" and "budget_tokens" in msg:
                print(f"[Warn] {model} rejected budget_tokens ({msg[:120]}...); retrying with adaptive thinking.")
                kwargs["thinking"] = {"type": "adaptive"}
                thinking_mode = "adaptive"
                r = client.messages.create(**kwargs)
            else:
                raise
        # Only text blocks form the reply; thinking / redacted_thinking blocks are
        # never parsed for JSON.
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        u = r.usage
        pt = getattr(u, "input_tokens", None)
        ct = getattr(u, "output_tokens", None)
        out_details = getattr(u, "output_tokens_details", None)
        thinking_tokens = getattr(out_details, "thinking_tokens", None) if out_details else None
        raw_usage = u.model_dump() if hasattr(u, "model_dump") else {}
        raw_usage["n_thinking_blocks"] = sum(
            1 for b in r.content if getattr(b, "type", "") in ("thinking", "redacted_thinking"))
        return LLMResponse(
            text=text, provider=provider, model_requested=model,
            model_returned=getattr(r, "model", model),
            latency_s=time.perf_counter() - t0,
            prompt_tokens=pt, completion_tokens=ct,
            # Anthropic bills thinking inside output_tokens; the split is only
            # available when usage.output_tokens_details.thinking_tokens is set.
            reasoning_tokens=thinking_tokens,
            cached_prompt_tokens=getattr(u, "cache_read_input_tokens", None),
            total_tokens=(pt + ct) if (pt is not None and ct is not None) else None,
            n_images=n_images,
            raw_usage=raw_usage,
            stop_reason=getattr(r, "stop_reason", None),
            thinking_budget=thinking_budget,
            thinking_mode=thinking_mode,
        )

    if provider == "google":
        from google.genai import types as gtypes

        system, rest = _split_system(messages)
        contents = []
        for m in rest:
            parts = []
            for p in _parts(m.get("content", "")):
                if p.get("type") == "text":
                    parts.append(gtypes.Part.from_text(text=p["text"]))
                elif p.get("type") == "image_url":
                    import base64 as _b64
                    media, data = _b64_from_data_url(p["image_url"]["url"])
                    parts.append(gtypes.Part.from_bytes(data=_b64.b64decode(data), mime_type=media))
            contents.append(gtypes.Content(role="user" if m.get("role") != "model" else "model",
                                           parts=parts))
        cfg: Dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        if temperature is not None:
            cfg["temperature"] = temperature
        r = _client(provider).models.generate_content(
            model=model, contents=contents, config=gtypes.GenerateContentConfig(**cfg))
        text = r.text or ""
        u = getattr(r, "usage_metadata", None)
        pt = getattr(u, "prompt_token_count", None) if u else None
        ct = getattr(u, "candidates_token_count", None) if u else None
        g_cands = getattr(r, "candidates", None) or []
        g_stop = getattr(g_cands[0], "finish_reason", None) if g_cands else None
        return LLMResponse(
            stop_reason=str(g_stop) if g_stop is not None else None,
            text=text, provider=provider, model_requested=model,
            model_returned=getattr(r, "model_version", model),
            latency_s=time.perf_counter() - t0,
            prompt_tokens=pt, completion_tokens=ct,
            reasoning_tokens=getattr(u, "thoughts_token_count", None) if u else None,
            cached_prompt_tokens=getattr(u, "cached_content_token_count", None) if u else None,
            total_tokens=getattr(u, "total_token_count", None) if u else None,
            n_images=n_images,
            raw_usage=u.model_dump() if (u is not None and hasattr(u, "model_dump")) else {},
        )

    raise ValueError(f"Unknown provider {provider!r}")


__all__ = ["chat", "LLMResponse", "infer_provider", "get_client"]
