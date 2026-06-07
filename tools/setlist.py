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
import unicodedata
import urllib.parse
from typing import Any

import requests

import config
import llm_client
import textfmt
from logging_config import get_logger
from prompts.setlist import build_setlist_prompt

logger = get_logger(__name__)

_DEEZER_BASE = "https://api.deezer.com"
_YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
_HTTP_TIMEOUT = 8

# Deezer public limit is 50 req / 5 sec. An 8-track set is 8-16 calls,
# well under, but a tiny inter-call pause keeps us safe under bursts.
_DEEZER_PAUSE_SECONDS = 0.05

# In-memory caches keyed by normalised query string.
_DEEZER_CACHE: dict[str, list[dict[str, Any]] | None] = {}
_YOUTUBE_CACHE: dict[str, str | None] = {}  # query -> video_id

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


def _youtube_video_id(artist: str, title: str) -> str | None:
    """
    Fetch the top YouTube video ID for an artist+title pair via the Data API v3.
    Returns None when YOUTUBE_API_KEY is not set, the quota is exhausted, or no
    hit is found. Results are cached in-memory for the session.
    """
    if not config.YOUTUBE_API_KEY:
        return None

    query = f"{artist} {title}".strip()
    if query in _YOUTUBE_CACHE:
        return _YOUTUBE_CACHE[query]

    try:
        resp = requests.get(
            _YOUTUBE_SEARCH_API,
            params={
                "part": "id",
                "q": query,
                "type": "video",
                "maxResults": 1,
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items") or []
        video_id: str | None = items[0]["id"]["videoId"] if items else None
    except Exception as exc:
        logger.warning("youtube_api_failed", query=query[:80], error=str(exc)[:160])
        video_id = None

    _YOUTUBE_CACHE[query] = video_id
    return video_id


# Titles too generic to search: Deezer will return any track, usually wrong.
_GENERIC_TITLES = frozenset({
    "untitled", "unknown", "n/a", "na", "track", "no title", "notitle",
    "unnamed", "demo", "instrumental",
})


def _fold(s: str) -> str:
    """
    Fold a name to a comparable form: accents removed (Âme -> ame, Rødhåd ->
    rodhad), lowercased, punctuation flattened to spaces. Both sides of a
    comparison go through this, so even imperfect folding stays symmetric.
    """
    # Manual map for letters NFKD does not decompose to ASCII.
    s = s.translate(str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ß": "ss"}))
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", " ", ascii_str.lower()).strip()


def _artist_matches(requested: str, returned: str) -> bool:
    """
    Decide whether a Deezer-returned artist is the artist we asked for.

    Designed against two real failure modes:
      - "Âme" (fold "ame") must NOT match "Ameli Dot" (substring of a word), nor
        "The Ame Church International Mass Choir" (one coincidental short token
        inside a long name).
      - "Floating Points" must still match "Floating Points feat. X".

    Rules, in order: exact fold wins; shared words count only if they are the
    majority of BOTH names and at least one is 4+ characters; a substring match
    is allowed only when the shorter side is itself 4+ characters (so a real
    single-word artist like "Surgeon" still matches "Surgeon & Lady Starlight",
    but "ame" cannot match a long unrelated name).
    """
    req = _fold(requested)
    ret = _fold(returned)
    if not req or not ret:
        return False
    if req == ret:
        return True

    req_words, ret_words = set(req.split()), set(ret.split())
    shared = req_words & ret_words
    if shared:
        majority = len(shared) >= max(len(req_words), len(ret_words)) / 2
        substantial = any(len(w) >= 4 for w in shared) or shared == req_words == ret_words
        if majority and substantial:
            return True

    shorter, longer = sorted((req, ret), key=len)
    return len(shorter) >= 4 and shorter in longer


def _deezer_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search Deezer for a track. Returns up to `limit` hits (empty list on failure).
    Cached in-memory by normalised query.
    """
    key = query.strip().lower()
    if key in _DEEZER_CACHE:
        cached = _DEEZER_CACHE[key]
        return cached if cached is not None else []

    try:
        resp = requests.get(
            f"{_DEEZER_BASE}/search",
            params={"q": query, "limit": limit},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("deezer_search_failed", query=query[:80], error=str(exc))
        _DEEZER_CACHE[key] = None
        return []

    hits = body.get("data") or []
    _DEEZER_CACHE[key] = hits
    return hits


def _hit_to_media(hit: dict[str, Any], fallback: bool) -> dict[str, str | bool | None]:
    """Shape a Deezer hit into the media dict. Deezer is the source of truth:
    matched_artist/matched_title carry the REAL track so the UI can display
    exactly what plays, never the LLM's possibly-invented title."""
    return {
        "preview_url": hit.get("preview") or None,
        "deezer_url": hit.get("link") or None,
        "matched_artist": (hit.get("artist") or {}).get("name") or None,
        "matched_title": hit.get("title") or None,
        "deezer_fallback": fallback,
    }


_NO_MATCH: dict[str, str | bool | None] = {
    "preview_url": None, "deezer_url": None,
    "matched_artist": None, "matched_title": None, "deezer_fallback": False,
}


def _enrich_track_with_deezer(
    artist: str, title: str
) -> dict[str, str | bool | None]:
    """
    Resolve an artist+title to a REAL Deezer track. Deezer is the source of
    truth: the returned matched_artist/matched_title are what the UI displays,
    so the shown title always matches the preview and the links.

    Strategy:
      1. Skip exact search for generic titles ("Untitled", "Unknown", etc.).
      2. Precise advanced query: artist:"X" track:"Y". Falls back to free-text
         "X Y" if the precise query misses. Pick the first hit whose artist
         actually matches (folded, accent-aware comparison).
      3. If no exact track is found, fall back to the artist's top track so the
         user still hears that artist's real sound. deezer_fallback=True so the
         UI labels it honestly.
    """
    title_key = re.sub(r"[^a-z0-9]", "", title.lower())
    skip_exact = title_key in _GENERIC_TITLES or not title_key

    if not skip_exact:
        hits = _deezer_search(f'artist:"{artist}" track:"{title}"')
        if not hits:
            hits = _deezer_search(f"{artist} {title}")
        for hit in hits:
            if _artist_matches(artist, (hit.get("artist") or {}).get("name", "")):
                return _hit_to_media(hit, fallback=False)
        if hits:
            logger.info(
                "deezer_artist_mismatch",
                requested=artist[:40],
                returned=(hits[0].get("artist") or {}).get("name", "")[:40],
            )

    time.sleep(_DEEZER_PAUSE_SECONDS)
    artist_hits = _deezer_search(f'artist:"{artist}"') or _deezer_search(artist)
    for hit in artist_hits:
        if _artist_matches(artist, (hit.get("artist") or {}).get("name", "")):
            return _hit_to_media(hit, fallback=True)

    return dict(_NO_MATCH)


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
                    "deezer_fallback": bool, True when this is the artist's top
                        track because the exact title was not on Deezer,
                    "youtube_url": str, direct video link when confirmed via the
                        Data API, else a search link (always present),
                    "youtube_verified": bool, True when youtube_url is a direct
                        video link rather than a search,
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

        # Deezer is the source of truth: display the real track we found so the
        # title, preview, and links always agree. Fall back to the LLM's text
        # only when nothing was found at all.
        display_artist = str(media.get("matched_artist") or artist)
        display_title = str(media.get("matched_title") or track_title)

        # Smart YouTube link: a direct video link when the Data API confirms an
        # ID (using the real title), otherwise a search link. No iframe.
        video_id = _youtube_video_id(display_artist, display_title)
        if video_id:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        else:
            youtube_url = _youtube_search_url(display_artist, display_title)

        enriched_tracks.append({
            "artist": display_artist,
            "title": display_title,
            "reason": reason,
            "energy": energy,
            "preview_url": media["preview_url"],
            "deezer_url": media["deezer_url"],
            "deezer_fallback": bool(media.get("deezer_fallback")),
            "youtube_url": youtube_url,
            "youtube_verified": bool(video_id),
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
