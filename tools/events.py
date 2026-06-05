"""
Rave Atlas — find_events + compare_events tools.

find_events: queries Resident Advisor's unofficial GraphQL endpoint for
upcoming Berlin club events. Returns a normalised list — empty on any
API failure. Never invents events.

compare_events: takes a list of events and a taste profile, calls the LLM
with the COMPARE_PROMPT reasoning rubric, and returns them ranked with
plain-language explanations. Falls back to an empty ranked list on LLM
or JSON-parse failure rather than raising.

RA schema notes (confirmed via live introspection, June 2026):
  - query:       eventListings  (not listing — that query errors downstream)
  - filter type: FilterInputDtoInput  (single DTO, not [FilterInput] array)
  - response:    data { event { ... } }  (not data { ... on Event { ... } })
  - Berlin area ID: 34  (from areas(searchTerm: "Berlin").id)
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

import llm_client
from logging_config import get_logger
from prompts.compare import build_compare_prompt

logger = get_logger(__name__)

_RA_URL = "https://ra.co/graphql"
_RA_BERLIN_AREA_ID = 34  # confirmed: areas(searchTerm="Berlin").id

_RA_QUERY = """
query GET_BERLIN_EVENTS(
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

_RA_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://ra.co/events/de/berlin",
    "Origin": "https://ra.co",
}

_HTTP_TIMEOUT = 15


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


def _normalize_event(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one eventListings data row to a clean Rave Atlas event dict.
    Each row has shape { event: { ... } }.
    """
    raw = row.get("event") or {}
    venue_obj = raw.get("venue") or {}
    area_obj = venue_obj.get("area") or {}
    artists = [a["name"] for a in (raw.get("artists") or []) if a.get("name")]
    genres = [g["name"] for g in (raw.get("genres") or []) if g.get("name")]
    cost_str = raw.get("cost") or ""
    content_url = raw.get("contentUrl") or ""
    url = f"https://ra.co{content_url}" if content_url.startswith("/") else content_url

    return {
        "id": str(raw.get("id", "")),
        "name": raw.get("title") or "Unknown event",
        "date": (raw.get("date") or "")[:10],
        "start_time": raw.get("startTime") or "",
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
) -> list[dict[str, Any]]:
    """
    Fetch upcoming Berlin club events from Resident Advisor.

    CALL THIS TOOL when the user asks about live or upcoming Berlin events
    on specific dates ("this Friday", "tonight", "next weekend"). Translate
    relative dates to ISO format (YYYY-MM-DD) before calling.

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
                   "area"      (str)    — keep only events in this Berlin neighbourhood

    Returns:
        List of normalised event dicts. Empty list when RA is unreachable or
        returns no results. Never returns invented events.
    """
    filters = filters or {}

    payload = {
        "query": _RA_QUERY,
        "variables": {
            "filters": {
                "areas": {"eq": _RA_BERLIN_AREA_ID},
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
        logger.warning("ra_request_failed", error=str(exc), date_from=date_from, date_to=date_to)
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

    events = [_normalize_event(r) for r in rows]

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
