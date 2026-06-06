"""
Rave Atlas, central configuration.

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
# Three models, one per provider, chosen to demonstrate the multi-provider
# router (llm_client.py) without bloating the picker:
#
#   - Claude Haiku 4.5 (OpenRouter): the default. Fastest and cheapest tier,
#     and the one model that works on a default OpenRouter account without any
#     privacy-policy configuration, so a fresh clone runs out of the box.
#   - Mistral Large (Mistral): a capable non-Anthropic alternative. It reuses
#     the same MISTRAL_API_KEY that powers the safety moderation call, so
#     enabling it costs no extra credential.
#   - GPT-OSS 120B (Ollama Cloud): an open-weights model, proof the same
#     OpenAI-compatible client routes to a third provider unchanged. Needs
#     OLLAMA_API_KEY; absent that, the picker entry simply fails on selection
#     and the user falls back to Haiku.

AVAILABLE_MODELS: list[dict[str, str]] = [
    {
        "id": "anthropic/claude-haiku-4-5",
        "name": "Claude Haiku 4.5 (fast, default)",
        "provider": "openrouter",
    },
    {
        "id": "mistral-large-latest",
        "name": "Mistral Large",
        "provider": "mistral",
    },
    {
        "id": "gpt-oss:120b",
        "name": "GPT-OSS 120B (open source)",
        "provider": "ollama",
    },
]

# Default to Haiku 4.5: cheapest tier, fastest, and the model that reliably
# passes OpenRouter's data-policy guardrails on a default account. Override
# via DEFAULT_MODEL in .env (must be an id present in AVAILABLE_MODELS).
DEFAULT_MODEL: str = os.environ.get(
    "DEFAULT_MODEL", "anthropic/claude-haiku-4-5"
)

# Per-model price table: (prompt $/1k tokens, completion $/1k tokens).
# Kept here so llm_client.py can estimate cost without hard-coding prices in
# logic. Open-weights models on Ollama Cloud are billed per-account, not
# per-token, so they show as 0.0 here and the cost badge reads $0 for them.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (0.00025, 0.00125),
    "mistral-large-latest": (0.002, 0.006),
    "gpt-oss:120b": (0.0, 0.0),
}

# ── Cities ────────────────────────────────────────────────────────────────────
#
# Berlin is the app's *home* city, the knowledge base, persona, venue
# commentary, and scene history are all Berlin-grade. The events tool, however,
# is genuinely city-aware: find_events resolves any of these names to a
# Resident Advisor area ID live (see tools/events.py).
#
# This curated list is limited to cities RA actually covers well. Small towns
# (e.g. Aachen, Bonn) are deliberately excluded, RA's listings there are
# near-empty, and the honest move is not to offer a city we can't deliver.
# Cologne/Düsseldorf are the realistic picks for the western-Germany scene.

HOME_CITY: str = os.environ.get("HOME_CITY", "Berlin")

# European cities with meaningful Resident Advisor coverage (~100 cities).
# RA density is highest in Berlin, Amsterdam, London, Paris, Barcelona, Belgrade,
# and the major German cities. Smaller towns are excluded, RA's listings there
# are near-empty and the honest move is not to offer cities we can't deliver on.
# Coverage is checked live: tools/events.py resolve_area_id() returns None for
# unknown cities and the Explore tab shows an honest "no listings" message.
AVAILABLE_CITIES: list[str] = [
    # ── Germany ──────────────────────────────────────────────────────────────
    # Berlin sits at the end of its group on purpose: this is the "Beyond Berlin"
    # browse, and the Raves in Berlin agent tab is the better tool for Berlin.
    # Berlin stays in the list (a plain RA browse of it is still useful) but does
    # not lead the city picker.
    "Cologne",
    "Hamburg",
    "Frankfurt",
    "Munich",
    "Leipzig",
    "Düsseldorf",
    "Stuttgart",
    "Nuremberg",
    "Dresden",
    "Mannheim",
    "Hannover",
    "Berlin", # home city, full KB depth; better served by the Raves in Berlin tab
    # ── United Kingdom & Ireland ──────────────────────────────────────────────
    "London",
    "Manchester",
    "Bristol",
    "Edinburgh",
    "Glasgow",
    "Birmingham",
    "Leeds",
    "Dublin",
    # ── Austria & Switzerland ─────────────────────────────────────────────────
    "Vienna",
    "Zurich",
    "Geneva",
    "Basel",
    "Bern",
    "Graz",
    "Lausanne",
    # ── Netherlands, Belgium & Luxembourg ────────────────────────────────────
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "Eindhoven",
    "Brussels",
    "Ghent",
    "Antwerp",
    "Luxembourg City",
    # ── France ───────────────────────────────────────────────────────────────
    "Paris",
    "Lyon",
    "Marseille",
    "Bordeaux",
    "Toulouse",
    "Strasbourg",
    "Lille",
    "Nantes",
    "Nice",
    "Montpellier",
    # ── Iberia ───────────────────────────────────────────────────────────────
    "Barcelona",
    "Madrid",
    "Valencia",
    "Ibiza",
    "Seville",
    "Bilbao",
    "Palma",
    "Lisbon",
    "Porto",
    # ── Italy ────────────────────────────────────────────────────────────────
    "Milan",
    "Rome",
    "Bologna",
    "Florence",
    "Turin",
    "Naples",
    "Palermo",
    "Trieste",
    # ── Scandinavia & Iceland ─────────────────────────────────────────────────
    "Copenhagen",
    "Aarhus",
    "Stockholm",
    "Gothenburg",
    "Malmö",
    "Oslo",
    "Bergen",
    "Trondheim",
    "Helsinki",
    "Reykjavik",
    # ── Baltic States ─────────────────────────────────────────────────────────
    "Tallinn",
    "Riga",
    "Vilnius",
    # ── Central Europe ────────────────────────────────────────────────────────
    "Warsaw",
    "Kraków",
    "Wrocław",
    "Gdańsk",
    "Katowice",
    "Prague",
    "Brno",
    "Bratislava",
    "Budapest",
    # ── Balkans ───────────────────────────────────────────────────────────────
    "Belgrade",
    "Novi Sad",
    "Zagreb",
    "Split",
    "Ljubljana",
    "Sofia",
    "Plovdiv",
    "Bucharest",
    "Cluj-Napoca",
    # ── Greece ────────────────────────────────────────────────────────────────
    "Athens",
    "Thessaloniki",
    # ── Turkey ────────────────────────────────────────────────────────────────
    "Istanbul",
    "Izmir",
    # ── Caucasus & Eastern Europe ─────────────────────────────────────────────
    # Tbilisi: one of Europe's most talked-about underground scenes (Bassiani,
    # Khidi, Café Gallery). Kyiv: Closer, Mezzanine, and a thriving pre-war scene.
    "Tbilisi",
    "Kyiv",
    "Yerevan",
]

# ── City regions for the Explore tab filter ──────────────────────────────────
#
# Keys become option labels in the region selector.
# Empty list for "All Europe" is a sentinel, the tab shows all cities.
# Berlin is included (distinct from the Berlin agent: this is a plain RA browse).
# ~100 cities total across 14 regions.

CITY_REGIONS: dict[str, list[str]] = {
    "All Europe": [],
    "Germany": [
        "Cologne", "Hamburg", "Frankfurt", "Munich", "Leipzig", "Düsseldorf",
        "Stuttgart", "Nuremberg", "Dresden", "Mannheim", "Hannover", "Berlin",
    ],
    "United Kingdom & Ireland": [
        "London", "Manchester", "Bristol", "Edinburgh", "Glasgow",
        "Birmingham", "Leeds", "Dublin",
    ],
    "Austria & Switzerland": [
        "Vienna", "Graz", "Zurich", "Geneva", "Basel", "Bern", "Lausanne",
    ],
    "Netherlands, Belgium & Luxembourg": [
        "Amsterdam", "Rotterdam", "Utrecht", "Eindhoven",
        "Brussels", "Ghent", "Antwerp", "Luxembourg City",
    ],
    "France": [
        "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse",
        "Strasbourg", "Lille", "Nantes", "Nice", "Montpellier",
    ],
    "Iberia": [
        "Barcelona", "Madrid", "Valencia", "Ibiza", "Seville", "Bilbao", "Palma",
        "Lisbon", "Porto",
    ],
    "Italy": [
        "Milan", "Rome", "Bologna", "Florence", "Turin", "Naples",
        "Palermo", "Trieste",
    ],
    "Scandinavia & Iceland": [
        "Copenhagen", "Aarhus", "Stockholm", "Gothenburg", "Malmö",
        "Oslo", "Bergen", "Trondheim", "Helsinki", "Reykjavik",
    ],
    "Baltic States": [
        "Tallinn", "Riga", "Vilnius",
    ],
    "Central Europe": [
        "Warsaw", "Kraków", "Wrocław", "Gdańsk", "Katowice",
        "Prague", "Brno", "Bratislava", "Budapest",
    ],
    "Balkans": [
        "Belgrade", "Novi Sad", "Zagreb", "Split", "Ljubljana",
        "Sofia", "Plovdiv", "Bucharest", "Cluj-Napoca",
    ],
    "Greece": [
        "Athens", "Thessaloniki",
    ],
    "Turkey": [
        "Istanbul", "Izmir",
    ],
    "Caucasus & Eastern Europe": [
        "Tbilisi", "Kyiv", "Yerevan",
    ],
}


# ── External APIs ─────────────────────────────────────────────────────────────

DISCOGS_TOKEN: str = os.environ.get("DISCOGS_TOKEN", "")
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
# Optional. When set, web_search uses Brave Search (SLA, 2 k free/month) instead
# of the keyless DuckDuckGo fallback. Get a key at search.brave.com/api.
BRAVE_SEARCH_API_KEY: str = os.environ.get("BRAVE_SEARCH_API_KEY", "")
# Mistral's OpenAI-compatible endpoint. The same MISTRAL_API_KEY powers both the
# safety moderation call (safety.py) and Mistral Large as a selectable chat model.
MISTRAL_BASE_URL: str = os.environ.get(
    "MISTRAL_BASE_URL", "https://api.mistral.ai/v1"
)

# ── Telegram weekend digest (optional automation) ─────────────────────────────
#
# The Friday weekend digest is sent to Telegram by a standalone script
# (automation/weekend_telegram.py) run from a GitHub Actions cron, NOT from the
# Streamlit process, which can't be relied on to be awake on a schedule. Both
# values are optional: if either is missing, the sender logs and no-ops cleanly.

TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Observability: LangSmith ──────────────────────────────────────────────────

LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.environ.get("LANGSMITH_PROJECT", "berlin rave atlas")
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
    print(f"DEFAULT_MODEL : {DEFAULT_MODEL}")
    print(f"OPENROUTER_BASE_URL: {OPENROUTER_BASE_URL}")
    print(f"OLLAMA_BASE_URL : {OLLAMA_BASE_URL}")
    print()
    print("AVAILABLE_MODELS:")
    for m in AVAILABLE_MODELS:
        print(f" [{m['provider']:12s}] {m['id']} - {m['name']}")
    print()

    assert OPENROUTER_BASE_URL, "OPENROUTER_BASE_URL must not be empty"
    assert DISCOGS_TOKEN, "DISCOGS_TOKEN is missing, add it to .env"
    assert MISTRAL_API_KEY, "MISTRAL_API_KEY is missing, add it to .env"
    assert len(AVAILABLE_MODELS) > 0, "AVAILABLE_MODELS must not be empty"
    assert any(
        m["provider"] == "openrouter" for m in AVAILABLE_MODELS
    ), "At least one OpenRouter model required"
    assert DEFAULT_MODEL in {m["id"] for m in AVAILABLE_MODELS}, (
        f"DEFAULT_MODEL {DEFAULT_MODEL!r} must be one of AVAILABLE_MODELS"
    )
    assert all(
        m["id"] in MODEL_PRICES for m in AVAILABLE_MODELS
    ), "Every model in AVAILABLE_MODELS needs a MODEL_PRICES entry"

    print("All config assertions passed.")
