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
# Haiku 4.5 is the active model — fastest, cheapest, works out-of-the-box on
# default OpenRouter accounts without privacy-policy configuration.
#
# To re-enable additional models, uncomment the entries below and restart.
# Multi-model support is architecturally in place (see llm_client.py).

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "anthropic/claude-haiku-4.5",
        "name": "Claude Haiku 4.5 (fast)",
        "provider": "openrouter",
    },
    {
        "id": "mistral-large-latest",
        "name": "Mistral Large (alternative)",
        "provider": "mistral",
    },
    # -- uncomment to re-enable additional models --
    # {
    #     "id": "anthropic/claude-sonnet-4.6",
    #     "name": "Claude Sonnet 4.6",
    #     "provider": "openrouter",
    # },
    # {
    #     "id": "anthropic/claude-opus-4.7",
    #     "name": "Claude Opus 4.7 (quality)",
    #     "provider": "openrouter",
    # },
    # {
    #     "id": "openai/gpt-5.5",
    #     "name": "GPT-5.5 (OpenAI via OpenRouter)",
    #     "provider": "openrouter",
    # },
    # {
    #     "id": "gemma3:27b",
    #     "name": "Gemma 3 27B (open-source)",
    #     "provider": "ollama",
    # },
    # {
    #     "id": "gpt-oss:120b",
    #     "name": "GPT-OSS 120B (open-source)",
    #     "provider": "ollama",
    # },
]

# Default to Haiku 4.5: cheapest tier, fastest, and the model that reliably
# passes OpenRouter's data-policy guardrails on a default account. Sonnet
# and Opus stay in AVAILABLE_MODELS as picker options; users with a
# whitelisted OpenRouter privacy policy can override via DEFAULT_MODEL in .env.
DEFAULT_MODEL: str = os.environ.get(
    "DEFAULT_MODEL", "anthropic/claude-haiku-4.5"
)

# Per-model price table: (prompt $/1k tokens, completion $/1k tokens).
# Kept here so llm_client.py can estimate cost without hard-coding prices in logic.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4.6":  (0.003,   0.015),
    "anthropic/claude-haiku-4.5":   (0.00025, 0.00125),
    "anthropic/claude-opus-4.7":    (0.015,   0.075),
    "openai/gpt-5.5":               (0.015,   0.060),
    "mistral-large-latest":         (0.002,   0.006),
    "gemma3:27b":                   (0.0,     0.0),
    "gpt-oss:120b":                 (0.0,     0.0),
}

# ── Cities ────────────────────────────────────────────────────────────────────
#
# Berlin is the app's *home* city — the knowledge base, persona, venue
# commentary, and scene history are all Berlin-grade. The events tool, however,
# is genuinely city-aware: find_events resolves any of these names to a
# Resident Advisor area ID live (see tools/events.py).
#
# This curated list is limited to cities RA actually covers well. Small towns
# (e.g. Aachen, Bonn) are deliberately excluded — RA's listings there are
# near-empty, and the honest move is not to offer a city we can't deliver.
# Cologne/Düsseldorf are the realistic picks for the western-Germany scene.

HOME_CITY: str = os.environ.get("HOME_CITY", "Berlin")

# European cities with meaningful Resident Advisor coverage.
# Organised by region. RA's density varies: Berlin, Amsterdam, Paris, Zurich,
# Barcelona, Belgrade and the major German cities are reliable; smaller cities
# (Aachen, Bonn, Wrocław, etc.) often return thin or empty results and are
# excluded to avoid misleading the user. Coverage is checked live by
# tools/events.py → resolve_area_id(); an unknown city resolves to None and
# the tab shows an honest "no listings" message rather than a fallback.
AVAILABLE_CITIES: list[str] = [
    # ── Germany ──────────────────────────────────────────────────────────────
    "Berlin",       # home city — full KB depth
    "Hamburg",
    "Cologne",
    "Frankfurt",
    "Munich",
    "Leipzig",
    "Düsseldorf",
    "Stuttgart",
    "Nuremberg",
    "Dresden",
    "Mannheim",
    "Hannover",
    # ── Austria & Switzerland ─────────────────────────────────────────────────
    "Vienna",
    "Zurich",
    "Geneva",
    "Basel",
    "Graz",
    # ── Netherlands & Belgium ─────────────────────────────────────────────────
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "Eindhoven",
    "Brussels",
    "Ghent",
    "Antwerp",
    # ── France ───────────────────────────────────────────────────────────────
    "Paris",
    "Lyon",
    "Marseille",
    "Bordeaux",
    "Toulouse",
    "Strasbourg",
    "Lille",
    "Nantes",
    # ── Iberia ───────────────────────────────────────────────────────────────
    "Barcelona",
    "Madrid",
    "Valencia",
    "Ibiza",
    "Seville",
    "Lisbon",
    "Porto",
    # ── Italy ────────────────────────────────────────────────────────────────
    "Milan",
    "Rome",
    "Bologna",
    "Florence",
    "Turin",
    "Naples",
    # ── Scandinavia ──────────────────────────────────────────────────────────
    "Copenhagen",
    "Stockholm",
    "Gothenburg",
    "Malmö",
    "Oslo",
    "Bergen",
    "Helsinki",
    # ── Central & Eastern Europe ─────────────────────────────────────────────
    "Warsaw",
    "Kraków",
    "Gdańsk",
    "Prague",
    "Brno",
    "Budapest",
    "Bucharest",
    "Cluj-Napoca",
    "Belgrade",
    "Zagreb",
    "Ljubljana",
    "Vilnius",
    "Riga",
    "Tallinn",
    # ── Greece & Turkey ──────────────────────────────────────────────────────
    "Athens",
    "Thessaloniki",
    "Istanbul",
]

# ── City regions for the Explore tab filter ──────────────────────────────────
#
# Keys become option labels in the region selector.
# Empty list for "All Europe" is a sentinel — the tab shows all cities.
# Berlin is included (distinct from the Berlin agent: this is a plain RA browse).

CITY_REGIONS: dict[str, list[str]] = {
    "All Europe": [],
    "Germany": [
        "Berlin", "Hamburg", "Cologne", "Frankfurt", "Munich", "Leipzig",
        "Düsseldorf", "Stuttgart", "Nuremberg", "Dresden", "Mannheim", "Hannover",
    ],
    "Austria & Switzerland": [
        "Vienna", "Graz", "Zurich", "Geneva", "Basel",
    ],
    "Netherlands & Belgium": [
        "Amsterdam", "Rotterdam", "Utrecht", "Eindhoven",
        "Brussels", "Ghent", "Antwerp",
    ],
    "France": [
        "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
        "Strasbourg", "Lille", "Nantes",
    ],
    "Southern Europe": [
        "Barcelona", "Madrid", "Valencia", "Ibiza", "Seville",
        "Lisbon", "Porto",
        "Milan", "Rome", "Bologna", "Florence", "Turin", "Naples",
        "Athens", "Thessaloniki",
    ],
    "Scandinavia": [
        "Copenhagen", "Stockholm", "Gothenburg", "Malmö",
        "Oslo", "Bergen", "Helsinki",
    ],
    "Central & Eastern Europe": [
        "Warsaw", "Kraków", "Gdańsk", "Prague", "Brno",
        "Budapest", "Bucharest", "Cluj-Napoca",
        "Belgrade", "Zagreb", "Ljubljana",
        "Vilnius", "Riga", "Tallinn", "Istanbul",
    ],
}


# ── External APIs ─────────────────────────────────────────────────────────────

DISCOGS_TOKEN: str = os.environ.get("DISCOGS_TOKEN", "")
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
# Mistral's OpenAI-compatible endpoint. The same MISTRAL_API_KEY powers both the
# safety moderation call (safety.py) and Mistral Large as a selectable chat model.
MISTRAL_BASE_URL: str = os.environ.get(
    "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
)

# ── Telegram weekend digest (optional automation) ─────────────────────────────
#
# The Friday weekend digest is sent to Telegram by a standalone script
# (automation/weekend_telegram.py) run from a GitHub Actions cron — NOT from the
# Streamlit process, which can't be relied on to be awake on a schedule. Both
# values are optional: if either is missing, the sender logs and no-ops cleanly.

TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

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
    os.environ.get("MODERATION_THRESHOLD", "0.85")
)
# Raised from 0.7 to 0.85: the 0.7 default over-fired on legitimate harm-reduction
# questions ("do people do drugs at raves?", "is MDMA common?") that are explicitly
# in scope for a Berlin rave-culture guide. At 0.85 the classifier still blocks
# clear malicious intent while allowing cultural and harm-reduction queries.


if __name__ == "__main__":
    print(f"DEFAULT_MODEL     : {DEFAULT_MODEL}")
    print(f"OPENROUTER_BASE_URL: {OPENROUTER_BASE_URL}")
    print(f"OLLAMA_BASE_URL   : {OLLAMA_BASE_URL}")
    print()
    print("AVAILABLE_MODELS:")
    for m in AVAILABLE_MODELS:
        print(f"  [{m['provider']:12s}]  {m['id']}  -  {m['name']}")
    print()

    assert OPENROUTER_BASE_URL, "OPENROUTER_BASE_URL must not be empty"
    assert DISCOGS_TOKEN, "DISCOGS_TOKEN is missing — add it to .env"
    assert MISTRAL_API_KEY, "MISTRAL_API_KEY is missing — add it to .env"
    assert len(AVAILABLE_MODELS) > 0, "AVAILABLE_MODELS must not be empty"
    assert any(
        m["provider"] == "openrouter" for m in AVAILABLE_MODELS
    ), "At least one OpenRouter model required"
    # Ollama models are commented out by default — remove this assertion when
    # re-enabling open-source models:
    # assert any(m["provider"] == "ollama" for m in AVAILABLE_MODELS)

    print("All config assertions passed.")
