"""
Rave Atlas, LLM client.

Wraps three OpenAI-compatible providers behind a single chat() function:
  - OpenRouter Anthropic Claude models (Haiku, and optionally Sonnet/Opus)
  - Mistral Mistral Large (also reuses the moderation key)
  - Ollama Cloud open-source models (Gemma 3 27B, GPT-OSS 120B) when enabled

Provider is selected automatically from config.AVAILABLE_MODELS based on
the requested model ID, no if/elif sprawl at call sites.

Features:
  - In-memory cache: repeated identical calls skip the API entirely
  - Exponential-backoff retry (3 attempts) on transient errors
  - Cost estimation from config.MODEL_PRICES
  - structlog on every call: model, provider, latency, tokens, cost, cache hit
  - get_chat_model() returns a LangChain ChatOpenAI for the agent graph
"""

from __future__ import annotations

import hashlib
import json
import time

import openai
from langchain_openai import ChatOpenAI
from langsmith import traceable

import config
from logging_config import get_logger

logger = get_logger(__name__)

# ── Provider clients (lazily initialised, one per provider) ───────────────────
_clients: dict[str, openai.OpenAI] = {}

# ── In-memory response cache (session-scoped) ─────────────────────────────────
_cache: dict[str, dict] = {}

# ── Error → user-safe message map ─────────────────────────────────────────────
_ERROR_MESSAGES: dict[type, str] = {
    openai.AuthenticationError: "API key invalid or expired, check your .env file.",
    openai.RateLimitError: "The AI service is busy right now. Please try again in a moment.",
    openai.APIConnectionError: "Cannot reach the AI service, check your internet connection.",
    openai.APITimeoutError: "The AI service timed out. Please try again.",
    openai.BadRequestError: "The request was malformed, please rephrase your message.",
    openai.InternalServerError: "The AI service returned an error. Please try again shortly.",
    openai.NotFoundError: (
        "Model not available, check the model ID in config.py against your "
        "OpenRouter dashboard (openrouter.ai/models), or adjust your account's "
        "data policy at openrouter.ai/settings/privacy."
    ),
}

_RETRYABLE: tuple[type, ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _provider_for(model_id: str) -> str:
    """Return the provider name for a model ID, defaulting to 'openrouter'."""
    for m in config.AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m["provider"]
    return "openrouter"


def _get_client(provider: str) -> openai.OpenAI:
    """Return a cached OpenAI client for the given provider."""
    if provider not in _clients:
        if provider == "ollama":
            _clients[provider] = openai.OpenAI(
                api_key=config.OLLAMA_API_KEY or "ollama",
                base_url=config.OLLAMA_BASE_URL,
            )
        elif provider == "mistral":
            _clients[provider] = openai.OpenAI(
                api_key=config.MISTRAL_API_KEY,
                base_url=config.MISTRAL_BASE_URL,
            )
        else: # openrouter (default)
            _clients[provider] = openai.OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/rave-atlas",
                    "X-Title": "Rave Atlas",
                },
            )
    return _clients[provider]


def _cache_key(
    model: str,
    temperature: float,
    top_p: float,
    messages: list[dict],
) -> str:
    payload = json.dumps(
        {"model": model, "temperature": temperature, "top_p": top_p, "messages": messages},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = config.MODEL_PRICES.get(model_id, (0.0, 0.0))
    return (prompt_tokens / 1000) * price_in + (completion_tokens / 1000) * price_out


def _user_safe_error(exc: Exception) -> str:
    for exc_type, message in _ERROR_MESSAGES.items():
        if isinstance(exc, exc_type):
            return message
    return f"Unexpected error: {type(exc).__name__}. Please try again."


# ── Main chat function ────────────────────────────────────────────────────────

@traceable(
    run_type="llm",
    name="llm_client.chat",
    metadata={"component": "llm_client"},
)
def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> dict:
    """
    Send a chat request to the appropriate LLM provider.

    Provider is selected automatically from AVAILABLE_MODELS based on model ID.
    Identical requests within a session are served from cache.

    Args:
        messages: OpenAI-format message list, e.g.
                     [{"role": "user", "content": "what is techno?"}]
        model: Model ID from AVAILABLE_MODELS. Defaults to DEFAULT_MODEL.
        temperature: Sampling temperature (0-2). Default 0.7.
        top_p: Nucleus sampling probability. Default 1.0.

    Returns:
        {
            "text": str, the assistant's reply,
            "model": str, model ID used,
            "provider": str, "openrouter" or "ollama",
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            },
            "cost_estimate": float, estimated USD cost (0.0 for free models),
            "cached": bool, True if served from in-memory cache,
            "latency_ms": int, wall-clock time in milliseconds (0 if cached),
        }
    """
    model = model or config.DEFAULT_MODEL
    provider = _provider_for(model)
    key = _cache_key(model, temperature, top_p, messages)

    if key in _cache:
        logger.info("llm_cache_hit", model=model, provider=provider)
        cached = dict(_cache[key])
        cached["cached"] = True
        cached["latency_ms"] = 0
        return cached

    client = _get_client(provider)
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=model,
                messages=messages, # type: ignore[arg-type]
                temperature=temperature,
                top_p=top_p,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            cost = _estimate_cost(model, prompt_tokens, completion_tokens)
            text = response.choices[0].message.content or ""

            result: dict = {
                "text": text,
                "model": model,
                "provider": provider,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "cost_estimate": round(cost, 6),
                "cached": False,
                "latency_ms": latency_ms,
            }

            _cache[key] = result
            logger.info(
                "llm_call",
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6),
                latency_ms=latency_ms,
                cached=False,
            )
            return result

        except _RETRYABLE as exc:
            last_exc = exc
            if attempt < 2:
                wait = 2 ** attempt # 1s, 2s
                logger.warning(
                    "llm_retry",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(exc)[:120],
                )
                time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            break

    safe_msg = _user_safe_error(last_exc) if last_exc else "Unknown error."
    logger.error("llm_failed", model=model, error=str(last_exc)[:120])
    raise RuntimeError(safe_msg) from last_exc


# ── LangChain helper (used by agent.py) ───────────────────────────────────────

def get_chat_model(
    model_id: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI instance wired to the correct provider.

    Used by agent.py to build the LangGraph ReAct agent without duplicating
    provider-routing logic.
    """
    model_id = model_id or config.DEFAULT_MODEL
    provider = _provider_for(model_id)

    if provider == "ollama":
        return ChatOpenAI(
            model=model_id,
            api_key=config.OLLAMA_API_KEY or "ollama",
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            top_p=top_p,
        )
    if provider == "mistral":
        return ChatOpenAI(
            model=model_id,
            api_key=config.MISTRAL_API_KEY,
            base_url=config.MISTRAL_BASE_URL,
            temperature=temperature,
            top_p=top_p,
        )
    return ChatOpenAI(
        model=model_id,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        top_p=top_p,
        default_headers={
            "HTTP-Referer": "https://github.com/rave-atlas",
            "X-Title": "Rave Atlas",
        },
        # Pin each model family to a single upstream provider to keep tool-call
        # IDs consistent within a conversation. OpenRouter load-balances across
        # Anthropic/Bedrock/Vertex for Claude, and across OpenAI/Azure for GPT
        # models; IDs from different backends do not round-trip, causing 400s on
        # follow-up turns. Google and DeepSeek models route to a single canonical
        # provider already, so no pin is needed for them.
        extra_body=_provider_pin(model_id),
    )


def _provider_pin(model_id: str) -> dict | None:
    """Return an OpenRouter provider-pin body for models that need it."""
    if model_id.startswith("anthropic/"):
        return {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}
    if model_id.startswith("openai/"):
        return {"provider": {"order": ["openai"], "allow_fallbacks": False}}
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: live chat call, DEFAULT_MODEL")
    print("=" * 60)
    msgs = [{"role": "user", "content": "In one sentence: what city is considered the world capital of techno?"}]
    r1 = chat(msgs)
    print(f" model : {r1['model']}")
    print(f" provider : {r1['provider']}")
    print(f" text : {r1['text']}")
    print(f" usage : {r1['usage']}")
    print(f" cost_usd : ${r1['cost_estimate']}")
    print(f" latency : {r1['latency_ms']}ms")
    print(f" cached : {r1['cached']}")
    assert r1["cached"] is False, "FAIL: first call should not be cached"
    assert len(r1["text"]) > 0, "FAIL: response text should not be empty"
    assert r1["usage"]["total_tokens"] > 0, "FAIL: token count should be > 0"

    print()
    print("=" * 60)
    print("Test 2: identical call, should be served from cache")
    print("=" * 60)
    r2 = chat(msgs)
    print(f" cached : {r2['cached']}")
    print(f" latency : {r2['latency_ms']}ms")
    print(f" text : {r2['text']}")
    assert r2["cached"] is True, "FAIL: second identical call must be cached"
    assert r2["latency_ms"] == 0, "FAIL: cached call latency should be 0"
    assert r2["text"] == r1["text"], "FAIL: cached text must match original"

    print()
    print("=" * 60)
    print("Test 3: provider routing, Claude model → openrouter")
    print("=" * 60)
    provider = _provider_for("anthropic/claude-haiku-4.5")
    print(f" anthropic/claude-haiku-4.5 → provider: {provider}")
    assert provider == "openrouter", "FAIL: Claude model should route to openrouter"
    provider2 = _provider_for("gpt-oss:120b")
    print(f" gpt-oss:120b → provider: {provider2}")
    assert provider2 == "ollama", "FAIL: GPT-OSS model should route to ollama"

    print()
    print("All assertions passed.")
