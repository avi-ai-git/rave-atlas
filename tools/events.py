"""
Rave Atlas — find_events + compare_events tools.

find_events: queries Resident Advisor's unofficial GraphQL endpoint for
upcoming club events in a given city. The city is resolved to an RA area ID
at runtime via RA's own areas(searchTerm) query (cached in-process), so the
tool genuinely queries the requested city rather than silently defaulting to
Berlin. Returns a normalised list — empty on any API failure or when the city
is not covered by RA. Never invents events.

compare_events: takes a list of events and a taste profile, calls the LLM
with the COMPARE_PROMPT reasoning rubric, and returns them ranked with
plain-language explanations. Falls back to an empty ranked list on LLM
or JSON-parse failure rather than raising.

RA schema notes (confirmed via live introspection, June 2026):
  - events query: eventListings  (not listing — that query errors downstream)
  - filter type:  FilterInputDtoInput  (single DTO, not [FilterInput] array)
  - response:     data { event { ... } }  (not data { ... on Event { ... } })
  - areas query:  areas(searchTerm: "berlin", limit: 3) { id name country { name } }
  - Berlin area ID: 34  (seeded — the app's home city; others resolved live)

Multi-city note: RA covers major scenes well (Berlin, Hamburg, Cologne,
Amsterdam, London, Paris). Coverage for small towns is sparse-to-empty — the
tool returns [] honestly rather than faking results. See config.AVAILABLE_CITIES.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
import llm_client
from logging_config import get_logger
from prompts.compare import build_compare_prompt

logger = get_logger(__name__)

_RA_URL = "https://ra.co/graphql"

# Area-ID cache, seeded with the home city's known-good ID so the default path
# (and the offline test-suite) never needs a network round-trip to resolve it.
# Keyed by lower-cased city name. Other cities are filled in lazily by
# resolve_area_id() via RA's areas() query.
_AREA_ID_CACHE: dict[str, int] = {"berlin": 34}

# Extended area metadata: keyed by lower-cased city name, includes area ID
# and the country name as returned by RA's areas query.  Populated lazily
# alongside _AREA_ID_CACHE; used to resolve the correct currency code.
_AREA_META_CACHE: dict[str, dict[str, object]] = {
    "berlin": {"id": 34, "country": "Germany"},
}

# European non-euro currencies keyed by country name (as RA returns them).
# All other European countries use EUR; this table lists the exceptions.
_COUNTRY_CURRENCY: dict[str, str] = {
    "Denmark":        "DKK",
    "Sweden":         "SEK",
    "Norway":         "NOK",
    "Switzerland":    "CHF",
    "Liechtenstein":  "CHF",
    "Czech Republic": "CZK",
    "Czechia":        "CZK",
    "Poland":         "PLN",
    "Hungary":        "HUF",
    "Romania":        "RON",
    "Serbia":         "RSD",
    "Turkey":         "TRY",
    "Iceland":        "ISK",
    "Bulgaria":       "BGN",
    "Croatia":        "EUR",   # adopted EUR 2023
}

_RA_EVENTS_QUERY = """
query GET_EVENTS(
  $filters: FilterInputDtoInput
  $pageSize: Int
  $page: Int
) {
  eventListings(
    filters: $filters
    pageSize: $pageSize
    page: $page
  ) {
    data {
      event {
        id
        title
        date
        startTime
        contentUrl
        venue {
          name
          area {
            name
          }
        }
        artists {
          name
        }
        cost
        genres {
          name
        }
      }
    }
    totalResults
  }
}
""".strip()

_RA_AREAS_QUERY = """
query GET_AREAS($searchTerm: String!) {
  areas(searchTerm: $searchTerm, limit: 3) {
    id
    name
    country {
      name
    }
  }
}
""".strip()

_RA_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://ra.co/events",
    "Origin": "https://ra.co",
}

_HTTP_TIMEOUT = 15


# ── Area resolution ─────────────────────────────────────────────────────────────

def resolve_area_id(city: str) -> int | None:
    """
    Resolve a city name to its Resident Advisor area ID.

    Checks the in-process cache first (seeded with Berlin=34), then falls back
    to RA's own areas(searchTerm) query and caches the result. Returns None when
    RA is unreachable or has no area matching the name — the caller treats that
    as "no coverage" and returns an empty event list rather than guessing.

    Args:
        city: City name as a user would type it (e.g. "Berlin", "Cologne").

    Returns:
        The numeric RA area ID, or None if it cannot be resolved.
    """
    key = (city or "").strip().lower()
    if not key:
        return None
    if key in _AREA_ID_CACHE:
        return _AREA_ID_CACHE[key]

    payload = {
        "query": _RA_AREAS_QUERY,
        "variables": {"searchTerm": city.strip()},
    }
    try:
        resp = requests.post(
            _RA_URL, json=payload, headers=_RA_HEADERS, timeout=_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("ra_area_resolve_failed", city=city, error=str(exc))
        return None

    areas = ((body.get("data") or {}).get("areas")) or []
    if not areas:
        logger.info("ra_area_not_found", city=city)
        return None

    try:
        area_id = int(areas[0]["id"])
    except (KeyError, ValueError, TypeError):
        logger.warning("ra_area_bad_id", city=city, raw=areas[0] if areas else None)
        return None

    _AREA_ID_CACHE[key] = area_id
    country_name: str = ((areas[0].get("country") or {}).get("name") or "")
    _AREA_META_CACHE[key] = {"id": area_id, "country": country_name}
    logger.info(
        "ra_area_resolved",
        city=city,
        area_id=area_id,
        matched=areas[0].get("name"),
        country=country_name,
    )
    return area_id


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_price(cost: str | None) -> float | None:
    """Return the minimum numeric EUR value from a RA cost string, or None."""
    if not cost:
        return None
    lower = cost.lower().strip()
    if "free" in lower or "donation" in lower:
        return 0.0
    m = re.search(r"\d+(?:[.,]\d+)?", lower)
    return float(m.group().replace(",", ".")) if m else None


def _get_city_currency(city: str) -> str:
    """
    Return the ISO 4217 currency code for a given city.

    Looks up the country stored in _AREA_META_CACHE (populated lazily by
    resolve_area_id).  Falls back to "EUR" for unknown or euro-zone cities.
    """
    key = (city or "").strip().lower()
    meta = _AREA_META_CACHE.get(key) or {}
    country = str(meta.get("country") or "")
    return _COUNTRY_CURRENCY.get(country, "EUR")


def _annotate_price(cost_str: str, city: str) -> str:
    """
    Append a currency code to a bare-number price string when the city uses a
    non-EUR currency.

    RA returns prices as free-form strings ("150", "€15", "Free", "15-20€").
    For non-EUR cities the string is just a number with no symbol.  This
    function detects that case and appends the correct currency code so the UI
    can display "150 DKK" instead of just "150".

    If the string already contains a currency symbol or code, or is not a bare
    number, it is returned unchanged.
    """
    if not cost_str:
        return cost_str
    lower = cost_str.lower().strip()
    # Already has a symbol or word — leave it alone
    if any(ch in cost_str for ch in ("€", "$", "£", "kr", "zł", "Ft", "lei")):
        return cost_str
    known_codes = set(_COUNTRY_CURRENCY.values())
    if any(code.lower() in lower for code in known_codes):
        return cost_str
    if "free" in lower or "donation" in lower:
        return cost_str

    # Bare number (possibly with a range like "10-20") — prefix with currency
    currency = _get_city_currency(city)
    if currency == "EUR":
        return cost_str  # RA normally includes "€" for EUR; leave as-is
    # Append code after the number(s)
    return f"{cost_str.strip()} {currency}"


def _normalize_event(row: dict[str, Any], city: str = "") -> dict[str, Any]:
    """
    Convert one eventListings data row to a clean Rave Atlas event dict.
    Each row has shape { event: { ... } }. `city` is the queried city, stored
    on the event so the UI and digest can label which city a listing belongs to.
    """
    raw = row.get("event") or {}
    venue_obj = raw.get("venue") or {}
    area_obj = venue_obj.get("area") or {}
    artists = [a["name"] for a in (raw.get("artists") or []) if a.get("name")]
    genres = [g["name"] for g in (raw.get("genres") or []) if g.get("name")]
    cost_str = _annotate_price(raw.get("cost") or "", city)
    content_url = raw.get("contentUrl") or ""
    url = f"https://ra.co{content_url}" if content_url.startswith("/") else content_url

    return {
        "id": str(raw.get("id", "")),
        "name": raw.get("title") or "Unknown event",
        "date": (raw.get("date") or "")[:10],
        "start_time": raw.get("startTime") or "",
        "city": city,
        "venue": venue_obj.get("name") or "Unknown venue",
        "area": area_obj.get("name") or "",
        "lineup": artists,
        "genres": genres,
        "price": cost_str,
        "price_numeric": _parse_price(cost_str),
        "url": url,
    }


# ── public tools ──────────────────────────────────────────────────────────────

def find_events(
    date_from: str,
    date_to: str,
    filters: dict[str, Any] | None = None,
    city: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch upcoming club events from Resident Advisor for a given city.

    CALL THIS TOOL when the user asks about live or upcoming events on specific
    dates ("this Friday", "tonight", "next weekend"). Translate relative dates
    to ISO format (YYYY-MM-DD) before calling.

    DO NOT call this tool for:
    - Music theory or history questions  →  use explain_music instead
    - Artist label or release background →  use enrich_artist instead

    Args:
        date_from: Inclusive start date, ISO format (e.g. "2024-01-05").
        date_to:   Inclusive end date,   ISO format (e.g. "2024-01-07").
        filters:   Optional dict with any of:
                   "max_price" (float)  — drop events priced above this EUR value
                   "genres"    (list)   — keep only events matching these genre names
                   "venue"     (str)    — keep only events at this venue (partial match)
                   "area"      (str)    — keep only events in this neighbourhood
        city:      City to search (e.g. "Berlin", "Cologne", "Amsterdam").
                   Defaults to config.HOME_CITY ("Berlin"). RA covers major
                   scenes well; small towns may return nothing.

    Returns:
        List of normalised event dicts. Empty list when the city has no RA
        coverage, RA is unreachable, or there are no results. Never invents events.
    """
    filters = filters or {}
    city = (city or config.HOME_CITY).strip()

    area_id = resolve_area_id(city)
    if area_id is None:
        logger.info("find_events_no_area", city=city, date_from=date_from, date_to=date_to)
        return []

    payload = {
        "query": _RA_EVENTS_QUERY,
        "variables": {
            "filters": {
                "areas": {"eq": area_id},
                "listingDate": {"gte": date_from, "lte": date_to},
                "eventType": {"eq": 1},
            },
            "pageSize": 25,
            "page": 1,
        },
    }

    try:
        resp = requests.post(
            _RA_URL,
            json=payload,
            headers=_RA_HEADERS,
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("ra_request_failed", error=str(exc), city=city, date_from=date_from, date_to=date_to)
        return []

    try:
        body = resp.json()
    except ValueError as exc:
        logger.warning("ra_json_parse_failed", error=str(exc))
        return []

    if "errors" in body:
        logger.warning("ra_graphql_errors", errors=body["errors"])

    rows: list[dict] = (
        (body.get("data") or {})
        .get("eventListings", {})
        .get("data", [])
    ) or []

    events = [_normalize_event(r, city=city) for r in rows]

    # Client-side filters — RA's server doesn't expose all these natively
    max_price: float | None = filters.get("max_price")
    if max_price is not None:
        events = [
            e for e in events
            if e["price_numeric"] is None or e["price_numeric"] <= max_price
        ]

    wanted_genres: list[str] = [g.lower() for g in (filters.get("genres") or [])]
    if wanted_genres:
        events = [
            e for e in events
            if any(g.lower() in wanted_genres for g in e["genres"])
        ]

    venue_filter = (filters.get("venue") or "").lower()
    if venue_filter:
        events = [e for e in events if venue_filter in e["venue"].lower()]

    area_filter = (filters.get("area") or "").lower()
    if area_filter:
        events = [e for e in events if area_filter in e["area"].lower()]

    logger.info(
        "find_events_ok",
        city=city,
        area_id=area_id,
        date_from=date_from,
        date_to=date_to,
        returned=len(events),
        filters=bool(filters),
    )
    return events


def compare_events(
    events: list[dict[str, Any]],
    taste_profile: dict[str, Any],
) -> dict[str, Any]:
    """
    Rank a list of Berlin events against a user's taste profile.

    CALL THIS TOOL after find_events when the user asks which event fits
    them, or which one to choose ("which should I go to", "what fits my
    taste", "rank these for me"). Produces honest rankings — mismatches
    are flagged, not quietly dropped.

    DO NOT call this tool:
    - Without first calling find_events (this tool needs real event data)
    - With an empty events list

    Args:
        events:        List of event dicts as returned by find_events.
        taste_profile: The user's taste profile dict from memory.load_profile.
                       Pass {} for a new user — rankings are generated but less
                       personalised.

    Returns:
        {
            "ranked_events": [
                {
                    "rank":        int,
                    "event_name":  str,
                    "fit_summary": str,
                    "reasoning":   str,
                    "tradeoff":    str | null,
                },
                ...
            ]
        }
        Returns {"ranked_events": []} on LLM failure rather than raising.
    """
    if not events:
        logger.info("compare_events_empty_input")
        return {"ranked_events": []}

    prompt = build_compare_prompt(events, taste_profile)

    try:
        result = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_text = result["text"].strip()
    except Exception as exc:
        logger.warning("compare_events_llm_failed", error=str(exc))
        return {"ranked_events": []}

    # Strip markdown fences if the model added them despite the instruction
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text.strip())

    try:
        ranked = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("compare_events_json_failed", error=str(exc), snippet=raw_text[:200])
        return {"ranked_events": []}

    logger.info("compare_events_ok", n_ranked=len(ranked.get("ranked_events", [])))
    return ranked


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date, timedelta

    today = date.today()
    days_ahead = (4 - today.weekday()) % 7 or 7
    fri = today + timedelta(days=days_ahead)
    sun = fri + timedelta(days=2)
    date_from = fri.isoformat()
    date_to = sun.isoformat()

    print("=" * 60)
    print(f"Test 1: find_events Berlin  {date_from} -> {date_to}")
    print("=" * 60)
    events = find_events(date_from, date_to)
    print(f"  returned {len(events)} events")
    for e in events[:5]:
        print(
            f"  [{e['date']}] {e['name'][:38]:<38}  "
            f"@ {e['venue'][:20]:<20}  {e['price'] or 'n/a':>8}  "
            f"lineup={e['lineup'][:2]}"
        )
    assert isinstance(events, list), "FAIL: find_events must return a list"

    print()
    print("=" * 60)
    print("Test 2: price filter max_price=0 -- free events only")
    print("=" * 60)
    free = find_events(date_from, date_to, filters={"max_price": 0})
    print(f"  free events: {len(free)}")
    for e in free:
        assert e["price_numeric"] is None or e["price_numeric"] == 0.0, (
            f"FAIL: price filter leaked paid event '{e['name']}' at {e['price']}"
        )
    print("  price filter OK")

    if events:
        print()
        print("=" * 60)
        print("Test 3: compare_events with techno taste profile")
        print("=" * 60)
        taste = {
            "preferred_genres": ["Techno", "Minimal Techno"],
            "loved_artists": ["Ben Klock", "Marcel Dettmann"],
            "budget_ceiling": 20,
            "preferred_areas": ["Friedrichshain"],
        }
        ranked = compare_events(events[:6], taste)
        print(f"  ranked {len(ranked.get('ranked_events', []))} events")
        for r in ranked.get("ranked_events", [])[:3]:
            print(f"  #{r['rank']}  {r['event_name'][:38]:<38}  {r['fit_summary'][:55]}")
        assert "ranked_events" in ranked, "FAIL: compare_events must return ranked_events key"
        assert isinstance(ranked["ranked_events"], list), "FAIL: ranked_events must be a list"
    else:
        print("\n  (no events this weekend -- compare_events test skipped)")

    print()
    print("Test 4: compare_events empty input -> graceful empty result")
    empty_result = compare_events([], {})
    assert empty_result == {"ranked_events": []}, "FAIL: empty input must return empty ranked_events"
    print("  empty-input guard OK")

    print()
    print("All assertions passed.")
