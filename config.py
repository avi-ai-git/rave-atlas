"""
Rave Atlas — central configuration.

All settings are loaded from environment variables (python-dotenv).
No hard-coded API keys, model names, or base URLs exist anywhere in this codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR: Path = Path(__file__).parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── LLM: OpenRouter (Anthropic Claude models) ─────────────────────────────────

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# ── LLM: Ollama Cloud (open-source models) ────────────────────────────────────

OLLAMA_API_KEY: str = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")

# ── Curated model list ────────────────────────────────────────────────────────
#
# Two providers, five models. "provider" tells llm_client.py which base URL
# and API key to use — no if/elif sprawl at call sites.
#
# Anthropic via OpenRouter: quality + reliability for graded deliverables.
# Ollama Cloud: open-source options that satisfy optional task #9 (multi-model)
#               without turning the picker into a model zoo.

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "anthropic/claude-sonnet-4.6",
        "name": "Claude Sonnet 4.6",
        "provider": "openrouter",
    },
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude Haiku 4.5 (fast)",
        "provider": "openrouter",
    },
    {
        "id": "anthropic/claude-opus-4.7",
        "name": "Claude Opus 4.7 (quality)",
        "provider": "openrouter",
    },
    {
        "id": "openai/gpt-5.5",
        "name": "GPT-5.5 (OpenAI via OpenRouter)",
        "provider": "openrouter",
    },
    {
        "id": "gemma3:27b",
        "name": "Gemma 3 27B (open-source)",
        "provider": "ollama",
    },
    {
        "id": "gpt-oss:120b",
        "name": "GPT-OSS 120B (open-source)",
        "provider": "ollama",
    },
]

DEFAULT_MODEL: str = os.environ.get(
    "DEFAULT_MODEL", "anthropic/claude-sonnet-4.6"
)

# Per-model price table: (prompt $/1k tokens, completion $/1k tokens).
# Kept here so llm_client.py can estimate cost without hard-coding prices in logic.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.6":  (0.003,   0.015),
    "anthropic/claude-haiku-4.5":   (0.00025, 0.00125),
    "anthropic/claude-opus-4.7":    (0.015,   0.075),
    "openai/gpt-5.5":               (0.015,   0.060),
    "gemma3:27b":                   (0.0,     0.0),
    "gpt-oss:120b":                 (0.0,     0.0),
}

# ── External APIs ─────────────────────────────────────────────────────────────

DISCOGS_TOKEN: str = os.environ.get("DISCOGS_TOKEN", "")
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")

# ── Observability: LangSmith ──────────────────────────────────────────────────

LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "rave-atlas")
LANGCHAIN_TRACING_V2: bool = (
    os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
)

# ── Storage ───────────────────────────────────────────────────────────────────

CHROMA_DIR: str = os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma"))
SQLITE_PATH: str = os.environ.get("SQLITE_PATH", str(DATA_DIR / "rave_atlas.db"))

# ── Safety tunables ───────────────────────────────────────────────────────────

RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS: int = int(
    os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")
)
MAX_INPUT_LENGTH: int = int(os.environ.get("MAX_INPUT_LENGTH", "2000"))
MIN_INPUT_LENGTH: int = int(os.environ.get("MIN_INPUT_LENGTH", "3"))
MODERATION_THRESHOLD: float = float(
    os.environ.get("MODERATION_THRESHOLD", "0.7")
)


if __name__ == "__main__":
    print(f"DEFAULT_MODEL     : {DEFAULT_MODEL}")
    print(f"OPENROUTER_BASE_URL: {OPENROUTER_BASE_URL}")
    print(f"OLLAMA_BASE_URL   : {OLLAMA_BASE_URL}")
    print()
    print("AVAILABLE_MODELS:")
    for m in AVAILABLE_MODELS:
        print(f"  [{m['provider']:12s}]  {m['id']}  —  {m['name']}")
    print()

    assert OPENROUTER_BASE_URL, "OPENROUTER_BASE_URL must not be empty"
    assert DISCOGS_TOKEN, "DISCOGS_TOKEN is missing — add it to .env"
    assert MISTRAL_API_KEY, "MISTRAL_API_KEY is missing — add it to .env"
    assert len(AVAILABLE_MODELS) > 0, "AVAILABLE_MODELS must not be empty"
    assert any(
        m["provider"] == "openrouter" for m in AVAILABLE_MODELS
    ), "At least one OpenRouter model required"
    assert any(
        m["provider"] == "ollama" for m in AVAILABLE_MODELS
    ), "At least one Ollama Cloud model required"

    print("All config assertions passed.")
