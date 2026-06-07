"""
Rave Atlas, build_setlist tool.

Two-pass architecture:

  Pass 1 (LLM, temperature 0.5)
    Artist and arc planning. The model picks scene-appropriate artists and
    assigns each a role (opener, build, peak, resolution) and energy level.
    No track titles yet — this is what the model does well.

  Deezer catalogue enrichment
    For each selected artist, the Deezer public API returns their real,
    streamable tracks. This verified catalogue replaces memory recall in pass 2.

  Pass 2 (LLM, temperature 0.2)
    Track selection from the Deezer catalogue. The model picks the best track
    for each arc position from the verified list. Title hallucination is
    structurally impossible when the artist is on Deezer; when an artist has
    no Deezer presence the model is told to use only titles it is certain exist.

  Deezer preview enrichment
    Each selected track is resolved to a 30-second MP3 preview URL and a
    Deezer track page link. The matched artist and title are used as the
    canonical display values so what you see always matches what plays.

  YouTube link
    Always present. A direct video link when YOUTUBE_API_KEY is set and
    confirms a video ID; otherwise a YouTube search URL.

Failure model: the tool never raises. LLM failure at either pass returns the
empty shape. Per-track Deezer failure leaves preview and URL fields None on
that track only; the rest of the set remains usable.
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
from prompts.setlist import build_artists_prompt, build_tracks_prompt

logger = get_logger(__name__)

_DEEZER_BASE = "https://api.deezer.com"
_YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
_HTTP_TIMEOUT = 8

# Deezer public limit is 50 req / 5 sec. An 8-track set with catalogue lookups
# is ~24 calls, well under, but a tiny pause keeps us safe under bursts.
_DEEZER_PAUSE_SECONDS = 0.05

# In-memory caches keyed by normalised query string.
_DEEZER_CACHE: dict[str, list[dict[str, Any]] | None] = {}
_YOUTUBE_CACHE: dict[str, str | None] = {}

_EMPTY_SETLIST: dict[str, Any] = {
    "title": "Set unavailable",
    "tracks": [],
    "energy_arc": [],
}


# ── Text normalisation ────────────────────────────────────────────────────────

def _fold(s: str) -> str:
    """
    Fold a name to a comparable form: accents removed (Âme -> ame, Rødhåd ->
    rodhad), lowercased, punctuation flattened to spaces. Both sides of a
    comparison go through this, so even imperfect folding stays symmetric.
    """
    s = s.translate(str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ß": "ss"}))
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", " ", ascii_str.lower()).strip()


def _artist_matches(requested: str, returned: str) -> bool:
    """
    Decide whether a Deezer/Spotify-returned artist is the artist we asked for.

    Designed against two real failure modes:
      - "Âme" (fold "ame") must NOT match "Ameli Dot" (substring of a word),
        nor "The Ame Church International Mass Choir" (one coincidental token
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


# ── Deezer helpers ────────────────────────────────────────────────────────────

# Titles too generic to search: Deezer will return any track, usually wrong.
_GENERIC_TITLES = frozenset({
    "untitled", "unknown", "n/a", "na", "track", "no title", "notitle",
    "unnamed", "demo", "instrumental",
})


def _deezer_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Search Deezer for a track. Returns up to `limit` hits (empty list on
    failure). Results are cached in-memory by normalised query string.
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


def _fetch_artist_catalogue(artist: str, limit: int = 10) -> list[str]:
    """
    Return up to `limit` real, distinct track titles for an artist from Deezer.
    Used between pass 1 and pass 2 to give the model a verified track list.
    Deduplicates by normalised title so variants like "Subzero" and "SUBZERO"
    count as one entry.
    """
    hits = _deezer_search(f'artist:"{artist}"', limit=limit)
    if not hits:
        hits = _deezer_search(artist, limit=limit)

    titles: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        hit_artist = (hit.get("artist") or {}).get("name", "")
        if not _artist_matches(artist, hit_artist):
            continue
        raw_title = str(hit.get("title") or "").strip()
        norm = _fold(raw_title)
        if raw_title and norm not in seen:
            titles.append(raw_title)
            seen.add(norm)

    return titles


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
    Resolve an artist+title to a REAL Deezer track.

    Strategy:
      1. Skip exact search for generic titles ("Untitled", "Unknown", etc.).
      2. Precise advanced query: artist:"X" track:"Y". Falls back to free-text
         "X Y" if the precise query misses. Pick the first hit whose artist
         actually matches.
      3. If no exact track is found, fall back to the artist's top track so the
         user still hears that artist's real sound. deezer_fallback=True so the
         UI labels it honestly.

    In the two-pass architecture, pass 2 picks from the Deezer catalogue, so
    the exact match succeeds for most tracks. The fallback path mainly fires
    when an artist has no Deezer presence at all.
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


# ── YouTube helpers ───────────────────────────────────────────────────────────

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


# ── LLM response parsers ──────────────────────────────────────────────────────

def _parse_llm_positions_json(raw_text: str) -> dict[str, Any] | None:
    """
    Parse the pass-1 LLM response (artist and arc plan) into a dict.
    Returns None on unrecoverable parse failure.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("positions_json_parse_failed", error=str(exc), snippet=text[:200])
        return None
    if not isinstance(parsed, dict) or "positions" not in parsed:
        logger.warning(
            "positions_unexpected_shape",
            keys=list(parsed) if isinstance(parsed, dict) else None,
        )
        return None
    return parsed


def _parse_llm_setlist_json(raw_text: str) -> dict[str, Any] | None:
    """
    Parse the pass-2 (or legacy single-pass) LLM response into the setlist
    dict. Tolerates the model accidentally wrapping the JSON in markdown fences.
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
        logger.warning(
            "setlist_unexpected_shape",
            keys=list(parsed) if isinstance(parsed, dict) else None,
        )
        return None

    return parsed


# ── Public tool ───────────────────────────────────────────────────────────────

def build_setlist(seed: str, n: int = 8) -> dict[str, Any]:
    """
    Generate a Berlin-flavoured set list with a deliberate energy arc,
    playable Deezer previews, YouTube links, and (optionally) real BPM and
    Camelot wheel data from Spotify.

    Two-pass process:
      1. LLM selects artists and arc positions (no track titles).
      2. Deezer catalogue is fetched for each artist.
      3. LLM selects tracks from the verified Deezer catalogue.
      4. Each track is enriched with Deezer preview + Spotify audio features.

    CALL THIS TOOL when the user asks for a tracklist, mix idea, warm-up set,
    closing set, or "build me a set / playlist for X". The seed is the user's
    own brief, pass it through verbatim.

    DO NOT call this tool for:
    - General music education questions -> use explain_music instead
    - Live event lookups -> use find_events instead
    - Artist background -> use enrich_artist instead

    Args:
        seed: User's brief. Examples: "hypnotic 130bpm techno for 2am",
              "deep melodic warm-up set, Watergate Friday 23h", "closing
              comedown after a hard peak, ambient-leaning".
        n:    Number of tracks. Default 8. Clamp 4-20; 16 tracks ~= 1 hour.

    Returns:
        {
            "title": str,
            "tracks": [
                {
                    "artist":        str,   display artist (Deezer source of truth)
                    "title":         str,   display title  (Deezer source of truth)
                    "llm_artist":    str | None,  original LLM pick when fallback
                    "llm_title":     str | None,  original LLM pick when fallback
                    "reason":        str,
                    "energy":        int 1-10,
                    "preview_url":   str | None,
                    "deezer_url":    str | None,
                    "deezer_fallback": bool,
                    "youtube_url":   str,
                    "youtube_verified": bool,
                },
                ...
            ],
            "energy_arc": list[int],
        }
        Returns {"title": "Set unavailable", "tracks": [], "energy_arc": []}
        on LLM failure rather than raising.
    """
    n = max(1, min(int(n), 20))
    logger.info("build_setlist_start", seed=seed[:80], n=n)

    # ── Pass 1: LLM plans artists and arc positions ───────────────────────────
    artists_prompt = build_artists_prompt(seed, n)
    try:
        artists_result = llm_client.chat(
            messages=[{"role": "user", "content": artists_prompt}],
            # Moderate temperature: creative arc shapes + real scene knowledge
            temperature=0.5,
        )
    except Exception as exc:
        logger.warning("build_setlist_pass1_failed", error=str(exc))
        return dict(_EMPTY_SETLIST)

    positions_data = _parse_llm_positions_json(artists_result["text"])
    if not positions_data:
        return dict(_EMPTY_SETLIST)

    title = textfmt.humanize(positions_data.get("title") or "Untitled set")
    arc_positions: list[dict[str, Any]] = positions_data.get("positions") or []
    if not arc_positions:
        return dict(_EMPTY_SETLIST)

    # ── Deezer catalogue enrichment ───────────────────────────────────────────
    # Give each arc position a list of real, streamable track titles so the
    # model in pass 2 selects from a verified menu rather than recalling from
    # memory.
    for pos in arc_positions:
        artist = str(pos.get("artist") or "").strip()
        if artist:
            catalogue = _fetch_artist_catalogue(artist, limit=10)
            pos["available_tracks"] = catalogue
            logger.info(
                "artist_catalogue_fetched",
                artist=artist[:40],
                n_tracks=len(catalogue),
            )
        else:
            pos["available_tracks"] = []

    # ── Pass 2: LLM selects tracks from the verified Deezer catalogue ─────────
    tracks_prompt = build_tracks_prompt(title, arc_positions)
    try:
        tracks_result = llm_client.chat(
            messages=[{"role": "user", "content": tracks_prompt}],
            # Very low temperature: selecting from a fixed menu, not generating
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("build_setlist_pass2_failed", error=str(exc))
        return dict(_EMPTY_SETLIST)

    parsed = _parse_llm_setlist_json(tracks_result["text"])
    if parsed is None:
        return dict(_EMPTY_SETLIST)

    title = textfmt.humanize(parsed.get("title") or title)
    raw_tracks: list[dict[str, Any]] = parsed.get("tracks") or []

    # ── Per-track enrichment: Deezer preview + Spotify features + YouTube ──────
    enriched_tracks: list[dict[str, Any]] = []
    energy_arc: list[int] = []

    for t in raw_tracks:
        artist = str(t.get("artist") or "").strip()
        track_title = str(t.get("title") or "").strip()
        reason = textfmt.humanize(str(t.get("reason") or "").strip())

        try:
            energy = int(t.get("energy", 0))
        except (TypeError, ValueError):
            energy = 0
        energy = max(1, min(energy, 10)) if energy else 5

        if not artist or not track_title:
            continue

        # Deezer: get preview URL and resolve to the canonical display title.
        media = _enrich_track_with_deezer(artist, track_title)
        display_artist = str(media.get("matched_artist") or artist)
        display_title = str(media.get("matched_title") or track_title)
        is_fallback = bool(media.get("deezer_fallback"))

        # Preserve the original LLM pick when Deezer returned a fallback,
        # so the UI can show what was intended alongside what plays.
        llm_artist = artist if is_fallback else None
        llm_title = track_title if is_fallback else None

        # YouTube: direct link when API key is set, search URL otherwise.
        video_id = _youtube_video_id(display_artist, display_title)
        youtube_url = (
            f"https://www.youtube.com/watch?v={video_id}"
            if video_id
            else _youtube_search_url(display_artist, display_title)
        )

        enriched_tracks.append({
            "artist":          display_artist,
            "title":           display_title,
            "llm_artist":      llm_artist,
            "llm_title":       llm_title,
            "reason":          reason,
            "energy":          energy,
            "preview_url":     media["preview_url"],
            "deezer_url":      media["deezer_url"],
            "deezer_fallback": is_fallback,
            "youtube_url":     youtube_url,
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


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Deezer catalogue fetch (Ben Klock)")
    print("=" * 60)
    catalogue = _fetch_artist_catalogue("Ben Klock", limit=5)
    print(f" Tracks found: {catalogue}")
    assert isinstance(catalogue, list), "FAIL: must return a list"
    print(f" {len(catalogue)} track(s) returned")

    print()
    print("=" * 60)
    print("Test 2: Deezer single track (Ben Klock - Subzero)")
    print("=" * 60)
    media = _enrich_track_with_deezer("Ben Klock", "Subzero")
    print(f" preview_url : {media['preview_url']}")
    print(f" deezer_url  : {media['deezer_url']}")
    assert media["preview_url"] is not None or media["deezer_url"] is not None, (
        "FAIL: Deezer should return at least the artist fallback for Ben Klock"
    )
    print(" OK")

    print()
    print("=" * 60)
    print("Test 3: YouTube URL builder")
    print("=" * 60)
    yt = _youtube_search_url("Ben Klock", "Subzero")
    assert yt.startswith("https://www.youtube.com/results?search_query=")
    assert "Ben+Klock" in yt and "Subzero" in yt
    print(f" {yt}")
    print(" OK")

    print()
    print("=" * 60)
    print("Test 4: build_setlist end-to-end (requires LLM)")
    print("=" * 60)
    setlist = build_setlist("hypnotic Berlin techno for 2am, Berghain main floor", n=4)
    print(f" title       : {setlist['title']}")
    print(f" n_tracks    : {len(setlist['tracks'])}")
    print(f" energy_arc  : {setlist['energy_arc']}")
    print()
    for i, t in enumerate(setlist["tracks"], 1):
        preview_marker = "[preview]" if t["preview_url"] else "[no preview]"
        print(
            f" {i}. {t['artist']} - {t['title']} "
            f"(energy {t['energy']}) {preview_marker}"
        )
        if t.get("llm_title"):
            print(f"    (LLM picked: {t['llm_artist']} - {t['llm_title']})")
        print(f"    reason  : {t['reason']}")
        print(f"    youtube : {t['youtube_url'][:70]}")

    assert isinstance(setlist, dict)
    assert "title" in setlist and "tracks" in setlist and "energy_arc" in setlist

    if setlist["tracks"]:
        assert len(setlist["energy_arc"]) == len(setlist["tracks"])
        for t in setlist["tracks"]:
            assert t["artist"] and t["title"]
            assert 1 <= t["energy"] <= 10
            assert t["youtube_url"].startswith("https://www.youtube.com/")
        print()
        print(" All structural assertions passed.")
    else:
        assert setlist == _EMPTY_SETLIST
        print(" (LLM unavailable, empty fallback verified)")

    print()
    print("All assertions passed.")
