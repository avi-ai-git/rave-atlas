"""
Tests for build_setlist tool.

Strategy:
  - Mock llm_client.chat to return controlled JSON payloads (no LLM dependency).
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

def _llm_response(title: str = "Test Set", tracks: list[dict] | None = None) -> dict:
    if tracks is None:
        tracks = [
            {"artist": "Ben Klock",        "title": "Subzero",   "reason": "Opening groover", "energy": 5},
            {"artist": "Marcel Dettmann",   "title": "Seduction", "reason": "Peak driver",     "energy": 8},
        ]
    payload = {
        "title": title,
        "tracks": tracks,
        "energy_arc": [t["energy"] for t in tracks],
    }
    return {"text": json.dumps(payload), "usage": {}, "cost": 0.0}


def _deezer_hit() -> Mock:
    resp = Mock()
    resp.json.return_value = {"data": [{
        "preview": "https://cdns-preview.dzcdn.net/stream/c-123.mp3",
        "link": "https://www.deezer.com/track/123",
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
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("hypnotic techno 2am", n=2)
        for key in ("title", "tracks", "energy_arc"):
            assert key in result

    def test_track_count_matches_llm_output(self, mocker):
        tracks = [
            {"artist": f"Artist {i}", "title": f"Track {i}", "reason": "...", "energy": 5}
            for i in range(4)
        ]
        mocker.patch("llm_client.chat", return_value=_llm_response(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("deep house warm-up", n=4)
        assert len(result["tracks"]) == 4

    def test_energy_arc_length_matches_tracks(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["energy_arc"]) == len(result["tracks"])

    def test_track_has_all_required_fields(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        for t in result["tracks"]:
            for key in ("artist", "title", "reason", "energy", "preview_url", "deezer_url", "youtube_url"):
                assert key in t, f"Missing key {key!r} in track"

    def test_deezer_hit_sets_preview_url(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_hit())
        result = setlist_mod.build_setlist("techno", n=2)
        assert any(t["preview_url"] is not None for t in result["tracks"])

    def test_deezer_miss_gives_none_preview_but_track_present(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
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

    def test_invalid_json_returns_empty_setlist(self, mocker):
        mocker.patch("llm_client.chat", return_value={"text": "not valid json at all", "usage": {}})
        result = setlist_mod.build_setlist("techno", n=2)
        assert result["tracks"] == []

    def test_energy_clamped_between_1_and_10(self, mocker):
        tracks = [
            {"artist": "A", "title": "T1", "reason": "r", "energy": 0},    # below min
            {"artist": "B", "title": "T2", "reason": "r", "energy": 11},   # above max
        ]
        mocker.patch("llm_client.chat", return_value=_llm_response(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        for t in result["tracks"]:
            assert 1 <= t["energy"] <= 10

    def test_malformed_track_without_artist_skipped(self, mocker):
        tracks = [
            {"artist": "",         "title": "Ghost Track", "reason": "...", "energy": 5},
            {"artist": "Ben Klock","title": "Subzero",     "reason": "ok",  "energy": 7},
        ]
        mocker.patch("llm_client.chat", return_value=_llm_response(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["artist"] == "Ben Klock"

    def test_malformed_track_without_title_skipped(self, mocker):
        tracks = [
            {"artist": "Ben Klock", "title": "",        "reason": "...", "energy": 5},
            {"artist": "Dettmann",  "title": "Seduction","reason": "ok", "energy": 8},
        ]
        mocker.patch("llm_client.chat", return_value=_llm_response(tracks=tracks))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        assert len(result["tracks"]) == 1
        assert result["tracks"][0]["artist"] == "Dettmann"

    def test_markdown_fenced_json_is_parsed(self, mocker):
        """LLM wrapping JSON in ```json fences must still parse correctly."""
        raw_json = json.dumps({
            "title": "Fenced Set",
            "tracks": [{"artist": "Dettmann", "title": "Seduction", "reason": "r", "energy": 6}],
            "energy_arc": [6],
        })
        mocker.patch("llm_client.chat", return_value={"text": f"```json\n{raw_json}\n```", "usage": {}})
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=1)
        assert result["title"] == "Fenced Set"
        assert len(result["tracks"]) == 1

    def test_youtube_url_always_present(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=2)
        for t in result["tracks"]:
            assert t["youtube_url"].startswith("https://www.youtube.com/results")

    def test_n_clamped_above_12(self, mocker):
        """n=100 is clamped to 12 in the prompt; tracks present reflect LLM output."""
        tracks_12 = [
            {"artist": f"A{i}", "title": f"T{i}", "reason": "r", "energy": 5}
            for i in range(12)
        ]
        mocker.patch("llm_client.chat", return_value=_llm_response(tracks=tracks_12))
        mocker.patch("requests.get", return_value=_deezer_miss())
        result = setlist_mod.build_setlist("techno", n=100)
        assert len(result["tracks"]) == 12

    def test_deezer_url_present_on_hit(self, mocker):
        mocker.patch("llm_client.chat", return_value=_llm_response())
        mocker.patch("requests.get", return_value=_deezer_hit())
        result = setlist_mod.build_setlist("techno", n=2)
        assert any(t["deezer_url"] is not None for t in result["tracks"])
