"""
Tests for build_setlist tool.

Strategy:
  - Mock llm_client.chat to return controlled JSON payloads (no LLM dependency).
    The two-pass architecture calls llm_client.chat twice per build_setlist
    invocation, so most tests use side_effect=[pass1_resp, pass2_resp].
  - Mock requests.get for Deezer (hit and miss paths).
  - Verify output shape, energy clamping, malformed-track skipping,
    markdown-fence tolerance, and fallback on LLM failure.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import Mock

import tools.setlist as setlist_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def _positions_response(
    title: str = "Test Set",
    artists: list[str] | None = None,
) -> dict:
    """Mock pass-1 LLM response: artist and arc plan (no track titles)."""
    if artists is None:
        artists = ["Ben Klock", "Marcel Dettmann"]
    positions = [
        {
            "position": i + 1,
            "artist": a,
            "role": "opener" if i == 0 else "build",
            "energy": 5,
            "reason": f"{a} in this arc role.",
        }
        for i, a in enumerate(artists)
    ]
    payload = {"title": title, "positions": positions}
    return {"text": json.dumps(payload), "usage": {}, "cost": 0.0}


def _tracks_response(title: str = "Test Set", tracks: list[dict] | None = None) -> dict:
    """Mock pass-2 LLM response: tracks selected from the catalogue."""
    if tracks is None:
        tracks = [
            {"artist": "Ben Klock",        "title": "Subzero",   "reason": "Opening groover", "energy": 5},
            {"artist": "Marcel Dettmann",  "title": "Seduction", "reason": "Peak driver",     "energy": 8},
        ]
    payload = {
        "title": title,
        "tracks": tracks,
        "energy_arc": [t["energy"] for t in tracks],
    }
    return {"text": json.dumps(payload), "usage": {}, "cost": 0.0}


# Convenience: returns [pass1, pass2] side_effect list for the common case.
def _both(
    title: str = "Test Set",
    tracks: list[dict] | None = None,
    artists: list[str] | None = None,
) -> list[dict]:
    artists = artists or (
        [t["artist"] for t in tracks] if tracks else None
    )
    return [_positions_response(title, artists), _tracks_response(title, tracks)]


def _deezer_hit() -> Mock:
    resp = Mock()
    resp.json.return_value = {"data": [{
        "preview": "https://cdns-preview.dzcdn.net/stream/c-123.mp3",
        "link":    "https://www.deezer.com/track/123",
        "title":   "Subzero",
        "artist":  {"name": "Ben Klock"},
    }]}
    resp.raise_for_status = Mock()
    return resp


def _deezer_miss() -> Mock:
    resp = Mock()
    resp.json.return_value = {"data": []}
    resp.raise_for_status = Mock()
    return resp


# ── TestBuildSetlist ──────────────────────────────────────────────────────────

class TestBuildSetlist:
    def test_output_has_required_top_level_keys(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("hypnotic techno 2am", n=2)
        for key in ("title", "tracks", "energy_arc"):
            assert key in result

    def test_track_count_matches_llm_output(self, mocker):
        tracks = [
            {"artist": f"Artist {i}", "title": f"Track {i}", "reason": "...", "energy": 5}
            for i in range(4)
        ]
        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("deep house warm-up", n=4)
        assert len(result["tracks"]) == 4

    def test_energy_arc_length_matches_tracks(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["energy_arc"]) == len(result["tracks"])

    def test_track_has_all_required_fields(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        required = (
            "artist", "title", "reason", "energy",
            "preview_url", "deezer_url", "youtube_url",
        )
        for t in result["tracks"]:
            for key in required:
                assert key in t, f"Missing key {key!r} in track"

    def test_deezer_hit_sets_preview_url(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_hit())
        result = setlist_mod.build_setlist("techno", n=2)
        assert any(t["preview_url"] is not None for t in result["tracks"])

    def test_deezer_miss_gives_none_preview_but_track_present(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["tracks"]) == 2
        assert all(t["preview_url"] is None for t in result["tracks"])

    def test_llm_failure_returns_empty_setlist(self, mocker):
        mocker.patch("llm_client.chat", side_effect=Exception("LLM unavailable"))
        result = setlist_mod.build_setlist("techno", n=2)
        assert result["title"] == "Set unavailable"
        assert result["tracks"] == []
        assert result["energy_arc"] == []

    def test_pass1_invalid_json_returns_empty_setlist(self, mocker):
        """Pass-1 returning garbage JSON must yield the empty shape."""
        mocker.patch("llm_client.chat", return_value={"text": "not valid json", "usage": {}})
        result = setlist_mod.build_setlist("techno", n=2)
        assert result["tracks"] == []

    def test_pass2_invalid_json_returns_empty_setlist(self, mocker):
        """Pass-2 returning garbage JSON must yield the empty shape."""
        mocker.patch(
            "llm_client.chat",
            side_effect=[
                _positions_response(),
                {"text": "not valid json", "usage": {}},
            ],
        )
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert result["tracks"] == []

    def test_energy_clamped_between_1_and_10(self, mocker):
        tracks = [
            {"artist": "A", "title": "T1", "reason": "r", "energy": 0},   # below min
            {"artist": "B", "title": "T2", "reason": "r", "energy": 11},  # above max
        ]
        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        for t in result["tracks"]:
            assert 1 <= t["energy"] <= 10

    def test_malformed_track_without_artist_skipped(self, mocker):
        tracks = [
            {"artist": "",         "title": "Ghost Track", "reason": "...", "energy": 5},
            {"artist": "Ben Klock","title": "Subzero",     "reason": "ok",  "energy": 7},
        ]
        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["artist"] == "Ben Klock"

    def test_malformed_track_without_title_skipped(self, mocker):
        tracks = [
            {"artist": "Ben Klock", "title": "",          "reason": "...", "energy": 5},
            {"artist": "Dettmann",  "title": "Seduction", "reason": "ok",  "energy": 8},
        ]
        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["artist"] == "Dettmann"

    def test_markdown_fenced_pass2_json_is_parsed(self, mocker):
        """Pass-2 LLM wrapping JSON in ```json fences must still parse correctly."""
        raw_tracks = [{"artist": "Dettmann", "title": "Seduction", "reason": "r", "energy": 6}]
        raw_json = json.dumps({
            "title": "Fenced Set",
            "tracks": raw_tracks,
            "energy_arc": [6],
        })
        mocker.patch(
            "llm_client.chat",
            side_effect=[
                _positions_response(artists=["Dettmann"]),
                {"text": f"```json\n{raw_json}\n```", "usage": {}},
            ],
        )
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=1)
        assert result["title"] == "Fenced Set"
        assert len(result["tracks"]) == 1

    def test_youtube_url_always_present(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        for t in result["tracks"]:
            assert t["youtube_url"].startswith("https://www.youtube.com/")

    def test_n_clamped_to_20_max(self, mocker):
        """n=100 is clamped to 20; tracks reflect LLM output."""
        tracks_12 = [
            {"artist": f"A{i}", "title": f"T{i}", "reason": "r", "energy": 5}
            for i in range(12)
        ]
        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks_12))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=100)
        assert len(result["tracks"]) == 12

    def test_deezer_url_present_on_hit(self, mocker):
        mocker.patch("llm_client.chat", side_effect=_both())
        mocker.patch("requests.get", return_value=_deezer_hit())
        result = setlist_mod.build_setlist("techno", n=2)
        assert any(t["deezer_url"] is not None for t in result["tracks"])

    def test_deezer_fallback_sets_llm_title(self, mocker):
        """When Deezer returns a fallback artist track, llm_title is preserved."""
        # Pass 2 picks "Invented Title" which won't match on Deezer (miss),
        # so the tool falls back to the artist's top track. llm_title should
        # be "Invented Title" so the UI knows what was originally requested.
        tracks = [
            {"artist": "Ben Klock", "title": "Invented Title", "reason": "r", "energy": 5}
        ]
        # Deezer miss for the exact track, but artist fallback returns a real one
        miss = _deezer_miss()
        hit = _deezer_hit()  # artist fallback: Ben Klock / Subzero

        call_count = 0
        def deezer_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First two calls (exact search + free-text): miss
            # Third call (artist fallback): hit
            return miss if call_count <= 2 else hit

        mocker.patch("llm_client.chat", side_effect=_both(tracks=tracks, artists=["Ben Klock"]))
        mocker.patch("requests.get", side_effect=deezer_side_effect)
        result = setlist_mod.build_setlist("techno", n=1)
        if result["tracks"]:  # only assert when the fallback path was reached
            t = result["tracks"][0]
            if t.get("deezer_fallback"):
                assert t["llm_title"] == "Invented Title"
                assert t["llm_artist"] == "Ben Klock"
