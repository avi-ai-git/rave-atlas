"""
Rave Atlas, find_events + compare_events tools.

find_events: queries Resident Advisor's unofficial GraphQL endpoint for
upcoming club events in a given city. The city is resolved to an RA area ID
at runtime via RA's own areas(searchTerm) query (cached in-process), so the
tool genuinely queries the requested city rather than silently defaulting to
Berlin. Returns a normalised list, empty on any API failure or when the city
is not covered by RA. Never invents events.

compare_events: takes a list of events and a taste profile, calls the LLM
with the COMPARE_PROMPT reasoning guide, and returns them ranked with
plain-language explanations. Falls back to an empty ranked list on LLM
or JSON-parse failure rather than raising.

RA schema notes (confirmed via live introspection, June 2026):
  - events query: eventListings (not listing, that query errors downstream)
  - filter type: FilterInputDtoInput (single DTO, not [FilterInput] array)
  - response: data { event { ... } } (not data { ... on Event { ... } })
  - areas query: areas(searchTerm: "berlin", limit: 3) { id name country { name } }
  - Berlin area ID: 34 (seeded, the app's home city; others resolved live)

Multi-city note: RA covers major scenes well (Berlin, Hamburg, Cologne,
Amsterdam, London, Paris). Coverage for small towns is sparse-to-empty, the
tool returns [] honestly rather than faking results. See config.AVAILABLE_CITIES.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import config
import llm_client
import textfmt
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
# and the country name as returned by RA's areas query. Populated lazily
# alongside _AREA_ID_CACHE; used to resolve the correct currency code.
_AREA_META_CACHE: dict[str, dict[str, object]] = {
    "berlin": {"id": 34, "country": "Germany"},
}

# European non-euro currencies keyed by country name (as RA returns them).
# All other European countries use EUR; this table lists the exceptions.
_COUNTRY_CURRENCY: dict[str, str] = {
    "Denmark": "DKK",
    "Sweden": "SEK",
    "Norway": "NOK",
    "Switzerland": "CHF",
    "Liechtenstein": "CHF",
    "Czech Republic": "CZK",
    "Czechia": "CZK",
    "Poland": "PLN",
    "Hungary": "HUF",
    "Romania": "RON",
    "Serbia": "RSD",
    "Turkey": "TRY",
    "Iceland": "ISK",
    "Bulgaria": "BGN",
    "Croatia": "EUR", # adopted EUR 2023
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
        endTime
        contentUrl
        venue {
          name
          address
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
    RA is unreachable or has no area matching the name, the caller treats that
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

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_clock(iso_datetime: str | None) -> str:
    """
    Pull a readable HH:MM clock time out of a Resident Advisor timestamp.

    RA returns full ISO datetimes like "2026-06-06T22:30:00.000", which is
    exactly what used to leak into the UI as unreadable "00:00:00.000" noise.
    This returns "22:30", or "" when there is no usable time (which is also
    what RA sends for events with no announced start, e.g. a bare midnight).
    """
    if not iso_datetime:
        return ""
    # Real RA values are full ISO datetimes ("2026-06-06T22:30:00.000"); accept
    # a bare "22:30" too so the function is robust to either shape.
    part = iso_datetime.split("T", 1)[1] if "T" in iso_datetime else iso_datetime
    clock = part[:5] # "22:30:00.000" -> "22:30"
    # A bare "00:00" from RA almost always means "no time announced", not an
    # actual midnight start, so we treat it as unknown rather than show it.
    if clock in ("", "00:00"):
        return ""
    return clock if len(clock) == 5 and clock[2] == ":" else ""


def _fmt_date_label(iso_date: str | None) -> str:
    """Turn an ISO date ("2026-06-06") into a friendly "Sat 6 Jun", or ""."""
    if not iso_date:
        return ""
    try:
        from datetime import date as _date

        d = _date.fromisoformat(iso_date[:10])
    except (ValueError, TypeError):
        return ""
    return f"{_WEEKDAY_NAMES[d.weekday()]} {d.day} {_MONTH_NAMES[d.month]}"


def _clean_address(address: str | None) -> str:
    """
    Tidy a Resident Advisor venue address into one readable line.

    RA addresses are inconsistent: some are clean ("Am Flutgraben 2, 12435
    Berlin"), some are semicolon-delimited with a trailing country ("Prinzen-
    strasse 85; Friedrichshain-Kreuzberg; 10969 Berlin; Germany"). This
    normalises separators to commas, drops a trailing "Germany", and collapses
    repeated whitespace so the card always shows a single clean address line.
    """
    if not address or not address.strip():
        return ""
    parts = [p.strip() for p in re.split(r"[;,]", address) if p.strip()]
    if parts and parts[-1].lower() in ("germany", "deutschland"):
        parts.pop()
    return ", ".join(parts)


def _berlin_registry_address(venue: str) -> str:
    """
    Return the curated address for a known Berlin venue, or "".

    For Berlin we trust the hand-checked club registry over RA's free-form
    address field when the venue clearly matches a registered club (Berghain,
    Tresor, Sisyphos, and the rest). For everything else the caller falls back
    to RA's own address.
    """
    name = (venue or "").strip()
    if not name:
        return ""
    try:
        from tools.club_registry import get_club

        entry = get_club(name)
    except Exception:
        return ""
    if entry is None:
        return ""
    # Guard against the registry's loose contains-match firing on a short or
    # generic venue name: only trust it when the names genuinely line up.
    a, b = name.lower(), entry.name.lower()
    if a == b or a.startswith(b) or b.startswith(a):
        return _clean_address(entry.address)
    return ""


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
    resolve_area_id). Falls back to "EUR" for unknown or euro-zone cities.
    """
    key = (city or "").strip().lower()
    meta = _AREA_META_CACHE.get(key) or {}
    country = str(meta.get("country") or "")
    return _COUNTRY_CURRENCY.get(country, "EUR")


# Preferred display symbol per ISO code. Codes not listed fall back to the code.
_CURRENCY_SYMBOL: dict[str, str] = {
    "EUR": "€",
    "DKK": "kr", "SEK": "kr", "NOK": "kr", "ISK": "kr",
    "CHF": "CHF",
    "CZK": "Kč", "PLN": "zł", "HUF": "Ft", "RON": "lei",
    "RSD": "RSD", "TRY": "₺", "BGN": "lv",
}

# ISO codes whose symbol is written BEFORE the number (e.g. "€13", "₺120").
# Everything else is written after (e.g. "150 kr", "120 zł").
_SYMBOL_BEFORE: frozenset[str] = frozenset({"EUR", "TRY"})


def _annotate_price(cost_str: str, city: str) -> str:
    """
    Return a display price that always carries the city's currency.

    RA returns prices as free-form strings ("150", "13/16", "€15", "Free", "").
    Crucially RA does NOT always include a currency symbol, even for euro cities,
    so a Paris event can come back as the bare "13/16". This function makes the
    price unambiguous:

      - empty / missing -> "Price unlisted"
      - free / donation -> "Free"
      - already has a symbol -> trusted, returned unchanged
        or known ISO code
      - bare number / range -> prefixed or suffixed with the city's currency
                                (e.g. Paris "13/16" -> "€13/16",
                                 Copenhagen "150" -> "150 kr")
    """
    currency = _get_city_currency(city)
    symbol = _CURRENCY_SYMBOL.get(currency, currency)

    if not cost_str or not cost_str.strip():
        return "Price unlisted"

    lower = cost_str.lower().strip()
    if "free" in lower or "donation" in lower:
        return "Free"

    # Already labelled with a recognised symbol or ISO code: trust RA's string.
    symbol_chars = ("€", "$", "£", "kr", "zł", "Ft", "lei", "Kč", "lv", "₺", "CHF")
    if any(s in cost_str for s in symbol_chars):
        return cost_str.strip()
    known_codes = set(_COUNTRY_CURRENCY.values()) | {"EUR"}
    if any(re.search(rf"\b{re.escape(code.lower())}\b", lower) for code in known_codes):
        return cost_str.strip()

    # Needs at least one digit to be a price. A digit-less value is either a
    # bare currency symbol ("€"), a lone dash ("-"), or stray punctuation that
    # RA sometimes returns instead of a real price. Show "Price unlisted"
    # rather than a meaningless symbol; keep it only if it carries real words.
    if not re.search(r"\d", cost_str):
        return cost_str.strip() if re.search(r"[A-Za-z]", cost_str) else "Price unlisted"

    n = cost_str.strip()
    if currency in _SYMBOL_BEFORE:
        return f"{symbol}{n}" # e.g. "€13/16", "₺120"
    return f"{n} {symbol}" # e.g. "150 kr", "120 zł"


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

    iso_date = (raw.get("date") or "")[:10]
    venue_name = venue_obj.get("name") or "Unknown venue"

    # Address: for Berlin trust the curated registry on a confident match,
    # otherwise fall back to RA's own (cleaned) address. Other cities use RA.
    address = ""
    if city.strip().lower() == "berlin":
        address = _berlin_registry_address(venue_name)
    if not address:
        address = _clean_address(venue_obj.get("address"))

    start_clock = _fmt_clock(raw.get("startTime"))
    end_clock = _fmt_clock(raw.get("endTime"))
    if start_clock and end_clock:
        time_label = f"{start_clock} to {end_clock}"
    else:
        time_label = start_clock # may be "" when RA announced no time

    return {
        "id": str(raw.get("id", "")),
        "name": raw.get("title") or "Unknown event",
        "date": iso_date,
        "date_label": _fmt_date_label(iso_date),
        "start_time": start_clock,
        "end_time": end_clock,
        "time_label": time_label,
        "city": city,
        "venue": venue_name,
        "area": area_obj.get("name") or "",
        "address": address,
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
    - Music theory or history questions → use explain_music instead
    - Artist label or release background → use enrich_artist instead

    Args:
        date_from: Inclusive start date, ISO format (e.g. "2024-01-05").
        date_to: Inclusive end date, ISO format (e.g. "2024-01-07").
        filters: Optional dict with any of:
                   "max_price" (float), drop events priced above this EUR value
                   "genres" (list), keep only events matching these genre names
                   "venue" (str), keep only events at this venue (partial match)
                   "area" (str), keep only events in this neighbourhood
        city: City to search (e.g. "Berlin", "Cologne", "Amsterdam").
                   Defaults to config.HOME_CITY ("Berlin"). RA covers major
                   scenes well; small towns may return nothing.

    Returns:
        List of normalised event dicts. Empty list when the city has no RA
        coverage, RA is unreachable, or there are no results. Never invents events.
    """
    filters = filters or {}
    city = (city or config.HOME_CITY).strip()

    # Date sanity: surface (don't silently honour) an obviously-stale date_from,
    # e.g. a model that ignored today's date and queried a past/last-year range.
    try:
        from datetime import date as _date

        if _date.fromisoformat(date_from[:10]) < _date.today():
            logger.warning(
                "find_events_past_date",
                date_from=date_from,
                date_to=date_to,
                today=_date.today().isoformat(),
                note="queried a date in the past; the model may have mis-resolved a relative date",
            )
    except (ValueError, TypeError):
        logger.warning("find_events_bad_date", date_from=date_from, date_to=date_to)

    area_id = resolve_area_id(city)
    if area_id is None:
        logger.info("find_events_no_area", city=city, date_from=date_from, date_to=date_to)
        return []

    # Fetch more when a venue filter is set: venue filtering is client-side, so
    # a small pageSize means a venue at position 26+ is silently missing.
    page_size = 100 if filters.get("venue") else 50

    payload = {
        "query": _RA_EVENTS_QUERY,
        "variables": {
            "filters": {
                "areas": {"eq": area_id},
                "listingDate": {"gte": date_from, "lte": date_to},
                "eventType": {"eq": 1},
            },
            "pageSize": page_size,
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

    # Client-side filters, RA's server doesn't expose all these natively
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
    taste", "rank these for me"). Produces honest rankings, mismatches
    are flagged, not quietly dropped.

    DO NOT call this tool:
    - Without first calling find_events (this tool needs real event data)
    - With an empty events list

    Args:
        events: List of event dicts as returned by find_events.
        taste_profile: The user's taste profile dict from memory.load_profile.
                       Pass {} for a new user, rankings are generated but less
                       personalised.

    Returns:
        {
            "ranked_events": [
                {
                    "rank": int,
                    "event_name": str,
                    "fit_summary": str,
                    "reasoning": str,
                    "tradeoff": str | null,
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

    # Enforce house typography on the reader-facing reasoning fields.
    for r in ranked.get("ranked_events", []):
        for field in ("fit_summary", "reasoning", "tradeoff"):
            if isinstance(r.get(field), str):
                r[field] = textfmt.humanize(r[field])

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
    print(f"Test 1: find_events Berlin {date_from} -> {date_to}")
    print("=" * 60)
    events = find_events(date_from, date_to)
    print(f" returned {len(events)} events")
    for e in events[:5]:
        print(
            f" [{e['date']}] {e['name'][:38]:<38} "
            f"@ {e['venue'][:20]:<20} {e['price'] or 'n/a':>8} "
            f"lineup={e['lineup'][:2]}"
        )
    assert isinstance(events, list), "FAIL: find_events must return a list"

    print()
    print("=" * 60)
    print("Test 2: price filter max_price=0 -- free events only")
    print("=" * 60)
    free = find_events(date_from, date_to, filters={"max_price": 0})
    print(f" free events: {len(free)}")
    for e in free:
        assert e["price_numeric"] is None or e["price_numeric"] == 0.0, (
            f"FAIL: price filter leaked paid event '{e['name']}' at {e['price']}"
        )
    print(" price filter OK")

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
        print(f" ranked {len(ranked.get('ranked_events', []))} events")
        for r in ranked.get("ranked_events", [])[:3]:
            print(f" #{r['rank']} {r['event_name'][:38]:<38} {r['fit_summary'][:55]}")
        assert "ranked_events" in ranked, "FAIL: compare_events must return ranked_events key"
        assert isinstance(ranked["ranked_events"], list), "FAIL: ranked_events must be a list"
    else:
        print("\n (no events this weekend -- compare_events test skipped)")

    print()
    print("Test 4: compare_events empty input -> graceful empty result")
    empty_result = compare_events([], {})
    assert empty_result == {"ranked_events": []}, "FAIL: empty input must return empty ranked_events"
    print(" empty-input guard OK")

    print()
    print("All assertions passed.")
