"""
Rave Atlas, build_setlist tool.

Combines three sources into a single user-facing artefact:

  1. LLM call with SETLIST_PROMPT, picks real Berlin-scene artists,
     plausible track titles, an energy arc (1-10 per track), and a
     one-line reason per track explaining its role in the arc.

  2. Deezer public API, for each track, fetches a 30-second MP3
     preview URL so the set is *playable* in the browser, not just
     readable. Falls back to artist-only search when the exact
     {artist + title} query has no hits (the SETLIST_PROMPT permits
     plausible-but-fictional titles, so exact matches sometimes miss).

  3. YouTube search URL, always available, requires no API call,
     and gives the user a one-click path to the full track.

Failure model: the tool never raises. LLM failure returns a valid empty
shape. Per-track Deezer failure leaves preview_url=None on that track
only; the rest of the setlist remains usable.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from typing import Any

import requests

import llm_client
import textfmt
from logging_config import get_logger
from prompts.setlist import build_setlist_prompt

logger = get_logger(__name__)

_DEEZER_BASE = "https://api.deezer.com"
_HTTP_TIMEOUT = 8

# Deezer public limit is 50 req / 5 sec. An 8-track set is 8-16 calls,
# well under, but a tiny inter-call pause keeps us safe under bursts.
_DEEZER_PAUSE_SECONDS = 0.05

# In-memory cache: normalised search query → Deezer result dict (or None)
_DEEZER_CACHE: dict[str, dict[str, Any] | None] = {}

_EMPTY_SETLIST: dict[str, Any] = {
    "title": "Set unavailable",
    "tracks": [],
    "energy_arc": [],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _youtube_search_url(artist: str, title: str) -> str:
    """Build a YouTube search URL for an artist+title pair."""
    query = f"{artist} {title}".strip()
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"


def _deezer_search(query: str) -> dict[str, Any] | None:
    """
    Search Deezer for a track. Returns the top hit dict or None.
    Cached in-memory by normalised query.
    """
    key = query.strip().lower()
    if key in _DEEZER_CACHE:
        return _DEEZER_CACHE[key]

    try:
        resp = requests.get(
            f"{_DEEZER_BASE}/search",
            params={"q": query, "limit": 1},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("deezer_search_failed", query=query[:80], error=str(exc))
        _DEEZER_CACHE[key] = None
        return None

    hits = body.get("data") or []
    result = hits[0] if hits else None
    _DEEZER_CACHE[key] = result
    return result


def _enrich_track_with_deezer(
    artist: str, title: str
) -> dict[str, str | bool | None]:
    """
    Return {preview_url, deezer_url, deezer_fallback} for an artist+title pair.

    Strategy:
      1. Try the exact "{artist} {title}" query.
      2. If no hit (which happens when the LLM invented a plausible-but-fake
         title), fall back to searching by artist alone, the user still
         gets a real 30s sample of that artist's actual sound.

    deezer_fallback is True when the artist-only fallback was used, so the UI
    can label the preview clearly ("similar track by {artist}") instead of
    implying it is the exact track.
    """
    hit = _deezer_search(f"{artist} {title}")
    if hit:
        return {
            "preview_url": hit.get("preview") or None,
            "deezer_url": hit.get("link") or None,
            "deezer_fallback": False,
        }

    time.sleep(_DEEZER_PAUSE_SECONDS)
    hit = _deezer_search(artist)

    if not hit:
        return {"preview_url": None, "deezer_url": None, "deezer_fallback": False}

    return {
        "preview_url": hit.get("preview") or None,
        "deezer_url": hit.get("link") or None,
        "deezer_fallback": True,
    }


def _parse_llm_setlist_json(raw_text: str) -> dict[str, Any] | None:
    """
    Parse the LLM response into the setlist dict. Tolerates the model
    accidentally wrapping the JSON in markdown fences despite instructions.
    Returns None on unrecoverable parse failure.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("setlist_json_parse_failed", error=str(exc), snippet=text[:200])
        return None

    if not isinstance(parsed, dict) or "tracks" not in parsed:
        logger.warning("setlist_unexpected_shape", keys=list(parsed) if isinstance(parsed, dict) else None)
        return None

    return parsed


# ── public tool ───────────────────────────────────────────────────────────────

def build_setlist(seed: str, n: int = 8) -> dict[str, Any]:
    """
    Generate a Berlin-flavoured set list with an energy arc and playable previews.

    CALL THIS TOOL when the user asks for a tracklist, mix idea, warm-up set,
    closing set, or "build me a set / playlist for X". The seed is the user's
    own brief, pass it through verbatim, do not paraphrase ("hypnotic 2am
    techno" beats "techno set").

    DO NOT call this tool for:
    - General music education questions → use explain_music instead
    - Live event lookups → use find_events instead
    - Artist background → use enrich_artist instead

    Args:
        seed: User's brief in their own words, vibe, time of night, venue
              feel, BPM target. Examples: "hypnotic 130bpm techno for 2am",
              "deep melodic warm-up set, Watergate Friday 23h", "closing
              comedown after a hard peak, ambient-leaning".
        n: Number of tracks. Default 8. Keep between 4 and 20, shorter
              loses the arc, 16 tracks gives a full 1-hour set (4 min/track).

    Returns:
        {
            "title": str, short evocative title for the set,
            "tracks": [
                {
                    "artist": str,
                    "title": str,
                    "reason": str, why this track in this position,
                    "energy": int, 1 (ambient) to 10 (peak saturation),
                    "preview_url": str | None, Deezer 30s MP3, None if unavailable,
                    "deezer_url": str | None, Deezer track page, None if unavailable,
                    "youtube_url": str, YouTube search link (always present),
                },
                ...
            ],
            "energy_arc": list[int], energy values in order (convenience),
        }

        Returns {"title": "Set unavailable", "tracks": [], "energy_arc": []}
        on LLM failure rather than raising.
    """
    n = max(1, min(int(n), 20)) # clamp, 20 is the upper bound; 16 is the 1h default
    prompt = build_setlist_prompt(seed, n)

    logger.info("build_setlist_start", seed=seed[:80], n=n)

    try:
        result = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
    except Exception as exc:
        logger.warning("build_setlist_llm_failed", error=str(exc))
        return dict(_EMPTY_SETLIST)

    parsed = _parse_llm_setlist_json(result["text"])
    if parsed is None:
        return dict(_EMPTY_SETLIST)

    title = textfmt.humanize(parsed.get("title") or "Untitled set")
    raw_tracks = parsed.get("tracks") or []

    enriched_tracks: list[dict[str, Any]] = []
    energy_arc: list[int] = []

    for t in raw_tracks:
        artist = str(t.get("artist") or "").strip()
        track_title = str(t.get("title") or "").strip()
        reason = textfmt.humanize(str(t.get("reason") or "").strip())

        # Energy may come back as int, float, or str, coerce defensively
        try:
            energy = int(t.get("energy", 0))
        except (TypeError, ValueError):
            energy = 0
        energy = max(1, min(energy, 10)) if energy else 5 # safe default

        if not artist or not track_title:
            continue # skip malformed tracks rather than emit garbage

        media = _enrich_track_with_deezer(artist, track_title)

        enriched_tracks.append({
            "artist": artist,
            "title": track_title,
            "reason": reason,
            "energy": energy,
            "preview_url": media["preview_url"],
            "deezer_url": media["deezer_url"],
            "deezer_fallback": bool(media.get("deezer_fallback")),
            "youtube_url": _youtube_search_url(artist, track_title),
        })
        energy_arc.append(energy)

    n_previews = sum(1 for t in enriched_tracks if t["preview_url"])
    logger.info(
        "build_setlist_ok",
        title=title[:80],
        n_tracks=len(enriched_tracks),
        n_previews=n_previews,
        energy_arc=energy_arc,
    )

    return {
        "title": title,
        "tracks": enriched_tracks,
        "energy_arc": energy_arc,
    }


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick offline tests first (don't depend on LLM availability)
    print("=" * 60)
    print("Test 0: YouTube URL builder")
    print("=" * 60)
    yt = _youtube_search_url("Ben Klock", "Subzero")
    print(f" {yt}")
    assert yt.startswith("https://www.youtube.com/results?search_query=")
    assert "Ben+Klock" in yt and "Subzero" in yt
    print(" OK")

    print()
    print("=" * 60)
    print("Test 1: Deezer search for a known track (Ben Klock - Subzero)")
    print("=" * 60)
    media = _enrich_track_with_deezer("Ben Klock", "Subzero")
    print(f" preview_url : {media['preview_url']}")
    print(f" deezer_url : {media['deezer_url']}")
    # Deezer should find at least *something* for Ben Klock
    assert media["preview_url"] is not None or media["deezer_url"] is not None, (
        "FAIL: Deezer should return at least the artist fallback for Ben Klock"
    )
    print(" Deezer returned a usable result")

    print()
    print("=" * 60)
    print("Test 2: build_setlist end-to-end (requires LLM)")
    print("=" * 60)
    setlist = build_setlist("hypnotic Berlin techno for 2am, Berghain main floor", n=6)
    print(f" title : {setlist['title']}")
    print(f" n_tracks : {len(setlist['tracks'])}")
    print(f" energy_arc : {setlist['energy_arc']}")
    print()
    for i, t in enumerate(setlist["tracks"], 1):
        preview_marker = "[preview]" if t["preview_url"] else "[no preview]"
        print(f" {i}. {t['artist']} - {t['title']} (energy {t['energy']}) {preview_marker}")
        print(f" reason : {t['reason']}")
        print(f" youtube : {t['youtube_url'][:70]}")

    assert isinstance(setlist, dict), "FAIL: build_setlist must return a dict"
    assert "title" in setlist and "tracks" in setlist and "energy_arc" in setlist, (
        "FAIL: setlist must have title, tracks, energy_arc keys"
    )

    if setlist["tracks"]: # only if LLM worked
        assert len(setlist["energy_arc"]) == len(setlist["tracks"]), (
            "FAIL: energy_arc length must match tracks length"
        )
        for t in setlist["tracks"]:
            assert t["artist"] and t["title"], "FAIL: every track needs artist + title"
            assert 1 <= t["energy"] <= 10, f"FAIL: energy out of range: {t['energy']}"
            assert t["youtube_url"].startswith("https://www.youtube.com/"), (
                "FAIL: youtube_url should always be present"
            )
        print()
        print(" All structural assertions passed.")
    else:
        print()
        print(" (LLM unavailable, structural tests on tracks skipped, "
              "but empty-shape fallback verified)")
        assert setlist == _EMPTY_SETLIST, "FAIL: failed LLM call must return empty fallback shape"

    print()
    print("All assertions passed.")
