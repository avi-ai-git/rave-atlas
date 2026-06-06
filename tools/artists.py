"""
Rave Atlas, enrich_artist tool.

Fetches label lineage, genre tags, and notable releases for an artist.
Discogs is the primary source (richest label data for electronic music);
MusicBrainz is the fallback for artists not covered by Discogs or when
the Discogs token is absent.

Used by the agent to explain why a specific lineup is worth attending, "this artist records on Klockworks, Ben Klock's own imprint" is grounded
and concrete rather than a vague genre assertion.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from config import DISCOGS_TOKEN
from logging_config import get_logger

logger = get_logger(__name__)

_DISCOGS_BASE = "https://api.discogs.com"
_MB_BASE = "https://musicbrainz.org/ws/2"

# MusicBrainz requires ≤1 unauthenticated request/sec
_MB_LAST_CALL: float = 0.0
_MB_MIN_INTERVAL: float = 1.1

# In-memory cache: normalised artist name → result dict
_ARTIST_CACHE: dict[str, dict[str, Any]] = {}

_HTTP_TIMEOUT: int = 10

# MusicBrainz requires a descriptive User-Agent
_MB_UA = "RaveAtlas/1.0 (contact@rave-atlas.example)"

_EMPTY: dict[str, Any] = {
    "name": "",
    "labels": [],
    "genres": [],
    "notable_releases": [],
    "summary_facts": [],
    "source": "none",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _throttle_mb() -> None:
    """Sleep if needed to keep MusicBrainz calls at ≤1/sec."""
    global _MB_LAST_CALL
    gap = _MB_MIN_INTERVAL - (time.monotonic() - _MB_LAST_CALL)
    if gap > 0:
        time.sleep(gap)
    _MB_LAST_CALL = time.monotonic()


def _discogs_headers() -> dict[str, str]:
    return {
        "User-Agent": "RaveAtlas/1.0",
        "Authorization": f"Discogs token={DISCOGS_TOKEN}",
    }


# ── Discogs ───────────────────────────────────────────────────────────────────

def _discogs_enrich(name: str) -> dict[str, Any] | None:
    if not DISCOGS_TOKEN:
        logger.warning("discogs_token_missing")
        return None

    hdrs = _discogs_headers()

    # 1. Search for artist
    try:
        sr = requests.get(
            f"{_DISCOGS_BASE}/database/search",
            params={"q": name, "type": "artist", "per_page": 5},
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        sr.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("discogs_search_error", artist=name, error=str(exc))
        return None

    results = sr.json().get("results", [])
    if not results:
        logger.info("discogs_not_found", artist=name)
        return None

    # Prefer exact name match; fall back to first result
    artist_id: int | None = None
    for r in results:
        if r.get("title", "").lower() == name.lower():
            artist_id = r["id"]
            break
    if artist_id is None:
        artist_id = results[0]["id"]

    # 2. Fetch artist detail (name + profile text)
    try:
        ar = requests.get(
            f"{_DISCOGS_BASE}/artists/{artist_id}",
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        ar.raise_for_status()
        artist_data = ar.json()
    except requests.RequestException as exc:
        logger.warning("discogs_artist_detail_error", artist_id=artist_id, error=str(exc))
        return None

    # 3. Fetch artist releases (labels + titles)
    try:
        rr = requests.get(
            f"{_DISCOGS_BASE}/artists/{artist_id}/releases",
            params={"sort": "year", "sort_order": "desc", "per_page": 10},
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        rr.raise_for_status()
        releases = rr.json().get("releases", [])
    except requests.RequestException as exc:
        logger.warning("discogs_releases_error", artist_id=artist_id, error=str(exc))
        releases = []

    labels: list[str] = []
    notable_releases: list[str] = []
    for rel in releases:
        label = rel.get("label")
        if label and label not in labels:
            labels.append(label)
        title = rel.get("title")
        year = rel.get("year")
        if title:
            entry = f"{title} ({year})" if year else title
            if entry not in notable_releases:
                notable_releases.append(entry)

    # 4. Release search for genre/style tags (best-effort; failure is OK)
    genres: list[str] = []
    try:
        gr = requests.get(
            f"{_DISCOGS_BASE}/database/search",
            params={"artist": name, "type": "release", "per_page": 5},
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        gr.raise_for_status()
        for rel in gr.json().get("results", []):
            for g in rel.get("genre", []):
                if g not in genres:
                    genres.append(g)
            for s in rel.get("style", []):
                if s not in genres:
                    genres.append(s)
    except requests.RequestException:
        pass # genres stay empty; caller will still get labels

    # Profile → first two sentences as summary facts
    profile = artist_data.get("profile", "").strip()
    summary_facts: list[str] = []
    if profile:
        sentences = [s.strip() for s in profile.replace("\n", " ").split(".") if len(s.strip()) > 10]
        summary_facts = sentences[:2]

    canonical = artist_data.get("name", name)
    logger.info(
        "discogs_enrich_ok",
        artist=canonical,
        labels=len(labels),
        genres=len(genres),
        releases=len(notable_releases),
    )

    return {
        "name": canonical,
        "labels": labels[:8],
        "genres": genres[:8],
        "notable_releases": notable_releases[:8],
        "summary_facts": summary_facts,
        "source": "discogs",
    }


# ── MusicBrainz ───────────────────────────────────────────────────────────────

def _musicbrainz_enrich(name: str) -> dict[str, Any] | None:
    hdrs = {"User-Agent": _MB_UA}

    # 1. Search artist
    _throttle_mb()
    try:
        sr = requests.get(
            f"{_MB_BASE}/artist",
            params={"query": name, "limit": 5, "fmt": "json"},
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        sr.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("mb_search_error", artist=name, error=str(exc))
        return None

    artists = sr.json().get("artists", [])
    if not artists:
        logger.info("mb_not_found", artist=name)
        return None

    # Prefer score ≥ 90; fall back to top result
    best = next((a for a in artists if int(a.get("score", 0)) >= 90), artists[0])
    mbid = best.get("id")
    if not mbid:
        return None

    # 2. Fetch artist detail with release-groups + tags
    _throttle_mb()
    try:
        dr = requests.get(
            f"{_MB_BASE}/artist/{mbid}",
            params={"inc": "release-groups+tags", "fmt": "json"},
            headers=hdrs,
            timeout=_HTTP_TIMEOUT,
        )
        dr.raise_for_status()
        detail = dr.json()
    except requests.RequestException as exc:
        logger.warning("mb_detail_error", mbid=mbid, error=str(exc))
        return None

    tags = detail.get("tags", [])
    genres = [t["name"] for t in sorted(tags, key=lambda t: -t.get("count", 0))[:8]]

    rgs = detail.get("release-groups", [])
    notable_releases: list[str] = []
    for rg in rgs[:8]:
        title = rg.get("title", "")
        year = (rg.get("first-release-date") or "")[:4]
        if title:
            notable_releases.append(f"{title} ({year})" if year else title)

    canonical = detail.get("name", name)
    disambiguation = detail.get("disambiguation", "")
    summary_facts = [disambiguation] if disambiguation else []

    logger.info("mb_enrich_ok", artist=canonical, genres=len(genres), releases=len(notable_releases))

    return {
        "name": canonical,
        "labels": [], # MB artist endpoint doesn't surface labels directly
        "genres": genres,
        "notable_releases": notable_releases,
        "summary_facts": summary_facts,
        "source": "musicbrainz",
    }


# ── Public tool ───────────────────────────────────────────────────────────────

def enrich_artist(name: str) -> dict[str, Any]:
    """
    Fetch label lineage, genre tags, and notable releases for a specific artist.

    CALL THIS TOOL when the user wants to understand a specific artist's:
    - Record labels / imprints they record on (e.g. Klockworks, Ostgut Ton)
    - Genre and style classification
    - Key or recent releases
    - Background context to decide if a lineup is worth attending

    DO NOT call this tool for:
    - Upcoming tour dates or live events → use find_events instead
    - General genre or scene questions → use explain_music instead

    Args:
        name: Artist name as it would appear on a release (e.g. "Ben Klock",
              "Aphex Twin", "Âme").

    Returns:
        {
            "name": str, canonical artist name from the data source,
            "labels": list, record labels / imprints the artist records on,
            "genres": list, genre and style tags,
            "notable_releases": list, recent / key release titles with year,
            "summary_facts": list, 1-2 sentence background facts,
            "source": str, "discogs", "musicbrainz", or "none"
        }
        Returns empty lists (not an exception) when the artist is not found
        or all APIs are unavailable.
    """
    cache_key = name.strip().lower()
    if cache_key in _ARTIST_CACHE:
        logger.info("enrich_artist_cache_hit", artist=name)
        return _ARTIST_CACHE[cache_key]

    logger.info("enrich_artist_start", artist=name)

    result = _discogs_enrich(name)

    if result is None:
        logger.info("falling_back_to_musicbrainz", artist=name)
        result = _musicbrainz_enrich(name)

    if result is None:
        logger.warning("enrich_artist_no_data", artist=name)
        result = {**_EMPTY, "name": name}

    _ARTIST_CACHE[cache_key] = result
    return result


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Test: enrich_artist('Ben Klock')")
    print("=" * 60)

    result = enrich_artist("Ben Klock")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()

    assert isinstance(result["labels"], list), "FAIL: labels must be a list"
    assert isinstance(result["genres"], list), "FAIL: genres must be a list"
    assert isinstance(result["notable_releases"], list), "FAIL: notable_releases must be a list"
    assert isinstance(result["summary_facts"], list), "FAIL: summary_facts must be a list"
    assert result["name"], "FAIL: name must not be empty"
    assert result["source"] in ("discogs", "musicbrainz", "none"), "FAIL: unexpected source value"

    print(f"source : {result['source']}")
    print(f"labels : {result['labels']}")
    print(f"genres : {result['genres']}")
    print(f"notable_releases : {result['notable_releases'][:3]}")

    # Cache hit test
    result2 = enrich_artist("Ben Klock")
    assert result2 is result, "FAIL: second call should return the cached object"
    print("\nCache hit confirmed.")

    # Empty-result test (artist that definitely does not exist)
    ghost = enrich_artist("ZZZNOBODYTHISARTISTDOESNOTEXIST999")
    assert ghost["labels"] == [], "FAIL: unknown artist should return empty labels"
    assert ghost["source"] in ("none", "musicbrainz"), "FAIL: unknown artist unexpected source"
    print(f"\nUnknown-artist result: name={ghost['name']!r}, source={ghost['source']!r}")

    print("\nAll assertions passed.")
