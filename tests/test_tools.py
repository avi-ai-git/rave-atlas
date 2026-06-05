"""
Tests for explain_music, enrich_artist, and find_events tools.

Strategy:
  - explain_music: mock ChromaDB collection; test grounded/ungrounded thresholds,
    allowlist filter passthrough, and graceful failure when DB is unavailable.
  - enrich_artist: mock requests.get with side_effect lists that simulate the 4
    sequential Discogs calls, MusicBrainz fallback, and both-miss path.
  - find_events: mock requests.post for RA GraphQL; test client-side filters,
    HTTP errors, and response normalisation.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock

import tools.music_kb as music_kb
import tools.artists as artists_mod
import tools.events as events_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_get_mock(json_body: dict, status_code: int = 200) -> Mock:
    resp = Mock()
    resp.json.return_value = json_body
    resp.raise_for_status = Mock()
    resp.status_code = status_code
    return resp


# ── TestExplainMusic ──────────────────────────────────────────────────────────

class TestExplainMusic:
    def _mock_collection(
        self,
        docs: list[str],
        distances: list[float],
        metadatas: list[dict] | None = None,
    ) -> Mock:
        if metadatas is None:
            metadatas = [{"source": "genres_techno.md", "doc_type": "genre"}] * len(docs)
        col = Mock()
        col.query.return_value = {
            "documents": [docs],
            "metadatas": [metadatas],
            "distances": [distances],
        }
        return col

    def test_grounded_below_threshold(self, mocker):
        col = self._mock_collection(
            docs=["Techno is a form of electronic dance music."],
            distances=[0.40],
        )
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        result = music_kb.explain_music("what is techno")
        assert result["grounded"] is True
        assert "Techno" in result["context"]
        assert len(result["sources"]) > 0

    def test_ungrounded_above_threshold(self, mocker):
        col = self._mock_collection(
            docs=["Tangentially related text."],
            distances=[0.75],
        )
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        result = music_kb.explain_music("what is the capital of France")
        assert result["grounded"] is False
        assert result["sources"] == []

    def test_at_threshold_boundary_is_ungrounded(self, mocker):
        """Distance exactly at SIMILARITY_THRESHOLD uses strict less-than → ungrounded."""
        col = self._mock_collection(docs=["Boundary."], distances=[music_kb.SIMILARITY_THRESHOLD])
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        result = music_kb.explain_music("boundary query")
        assert result["grounded"] is False

    def test_allowlist_filter_passed_to_collection_query(self, mocker):
        col = self._mock_collection(docs=["Genre content."], distances=[0.30])
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        music_kb.explain_music("what is minimal techno", allowed_doc_types=["genre"])
        where = col.query.call_args.kwargs.get("where")
        assert where == {"doc_type": {"$in": ["genre"]}}

    def test_no_allowlist_passes_none_filter(self, mocker):
        col = self._mock_collection(docs=["Some content."], distances=[0.30])
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        music_kb.explain_music("any query")
        where = col.query.call_args.kwargs.get("where")
        assert where is None

    def test_chroma_unavailable_returns_grounded_false(self, mocker):
        mocker.patch("tools.music_kb.get_collection", side_effect=Exception("DB unavailable"))
        result = music_kb.explain_music("any query")
        assert result["grounded"] is False
        assert result["sources"] == []
        assert "unavailable" in result["context"].lower()

    def test_empty_docs_returns_grounded_false(self, mocker):
        col = Mock()
        col.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        result = music_kb.explain_music("empty query")
        assert result["grounded"] is False
        assert result["sources"] == []

    def test_multiple_chunks_joined_with_separator(self, mocker):
        col = self._mock_collection(
            docs=["Chunk A.", "Chunk B."],
            distances=[0.30, 0.35],
            metadatas=[{"source": "genres_techno.md"}, {"source": "genres_house.md"}],
        )
        mocker.patch("tools.music_kb.get_collection", return_value=col)
        result = music_kb.explain_music("multi-chunk query")
        assert result["grounded"] is True
        assert "Chunk A." in result["context"]
        assert "Chunk B." in result["context"]


# ── TestEnrichArtist ──────────────────────────────────────────────────────────

class TestEnrichArtist:
    def _discogs_mocks(self) -> list[Mock]:
        """4 sequential requests.get responses for a successful Discogs lookup."""
        return [
            _make_get_mock({"results": [{"id": 42, "title": "Ben Klock"}]}),
            _make_get_mock({"name": "Ben Klock", "profile": "Ben Klock is a Berlin DJ. He records on Klockworks."}),
            _make_get_mock({"releases": [
                {"title": "Subzero", "year": 2009, "label": "Klockworks"},
                {"title": "One",     "year": 2012, "label": "Klockworks"},
            ]}),
            _make_get_mock({"results": [{"genre": ["Electronic"], "style": ["Techno", "Minimal Techno"]}]}),
        ]

    def test_discogs_success_returns_expected_shape(self, mocker):
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "fake-token")
        mocker.patch("requests.get", side_effect=self._discogs_mocks())
        result = artists_mod.enrich_artist("Ben Klock")
        assert result["name"] == "Ben Klock"
        assert "Klockworks" in result["labels"]
        assert any(g in result["genres"] for g in ("Techno", "Electronic"))
        assert any("Subzero" in r for r in result["notable_releases"])
        assert result["source"] == "discogs"

    def test_discogs_miss_falls_back_to_musicbrainz(self, mocker):
        """Discogs search returns empty → falls through to MusicBrainz."""
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "fake-token")
        mocker.patch("time.sleep")
        mocker.patch("requests.get", side_effect=[
            _make_get_mock({"results": []}),                                    # discogs search: miss
            _make_get_mock({"artists": [{"id": "mbid-001", "score": 95}]}),    # MB search
            _make_get_mock({                                                     # MB detail
                "name": "Ben Klock",
                "tags": [{"name": "techno", "count": 10}],
                "release-groups": [{"title": "One", "first-release-date": "2012"}],
                "disambiguation": "Berlin DJ",
            }),
        ])
        result = artists_mod.enrich_artist("Ben Klock Fallback")
        assert result["source"] == "musicbrainz"
        assert "techno" in result["genres"]

    def test_both_sources_miss_returns_empty_shape(self, mocker):
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "fake-token")
        mocker.patch("time.sleep")
        mocker.patch("requests.get", side_effect=[
            _make_get_mock({"results": []}),    # discogs miss
            _make_get_mock({"artists": []}),    # MB miss
        ])
        result = artists_mod.enrich_artist("ZZZNOBODYTHISARTISTDOESNOTEXIST999")
        assert result["labels"] == []
        assert result["genres"] == []
        assert result["source"] == "none"

    def test_cache_hit_skips_api_calls(self, mocker):
        """Second call with same name (case-insensitive) returns cache without new requests."""
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "fake-token")
        mock_get = mocker.patch("requests.get", side_effect=self._discogs_mocks())
        artists_mod.enrich_artist("Unique Cache Test Artist")
        count_after_first = mock_get.call_count
        artists_mod.enrich_artist("unique cache test artist")
        assert mock_get.call_count == count_after_first

    def test_no_discogs_token_skips_to_musicbrainz(self, mocker):
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "")
        mocker.patch("time.sleep")
        mocker.patch("requests.get", side_effect=[
            _make_get_mock({"artists": [{"id": "mbid-002", "score": 92}]}),
            _make_get_mock({
                "name": "Aphex Twin",
                "tags": [{"name": "idm", "count": 5}],
                "release-groups": [],
                "disambiguation": "",
            }),
        ])
        result = artists_mod.enrich_artist("Aphex Twin No Token")
        assert result["source"] == "musicbrainz"

    def test_result_always_has_all_required_keys(self, mocker):
        mocker.patch.object(artists_mod, "DISCOGS_TOKEN", "fake-token")
        mocker.patch("requests.get", side_effect=self._discogs_mocks())
        result = artists_mod.enrich_artist("Shape Check Artist")
        for key in ("name", "labels", "genres", "notable_releases", "summary_facts", "source"):
            assert key in result, f"Missing key: {key!r}"


# ── TestFindEvents ────────────────────────────────────────────────────────────

def _ra_response(event_rows: list[dict]) -> Mock:
    resp = Mock()
    resp.json.return_value = {
        "data": {
            "eventListings": {
                "data": event_rows,
                "totalResults": len(event_rows),
            }
        }
    }
    resp.raise_for_status = Mock()
    return resp


def _ra_row(
    title: str = "Test Night",
    price: str = "€15",
    genres: list[str] | None = None,
    venue: str = "Tresor",
    area: str = "Mitte",
    artists: list[str] | None = None,
) -> dict:
    return {
        "event": {
            "id": "1",
            "title": title,
            "date": "2024-01-05",
            "startTime": "23:00",
            "contentUrl": "/events/1",
            "venue": {"name": venue, "area": {"name": area}},
            "artists": [{"name": a} for a in (artists or ["DJ Test"])],
            "cost": price,
            "genres": [{"name": g} for g in (genres or ["Techno"])],
        }
    }


class TestFindEvents:
    def test_basic_success_returns_normalised_events(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([_ra_row(title="Berghain Friday")]))
        events = events_mod.find_events("2024-01-05", "2024-01-07")
        assert len(events) == 1
        assert events[0]["name"] == "Berghain Friday"
        assert events[0]["venue"] == "Tresor"
        assert events[0]["price"] == "€15"
        assert "Techno" in events[0]["genres"]

    def test_max_price_filter_excludes_expensive_events(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([
            _ra_row(title="Cheap Night", price="€10"),
            _ra_row(title="Expensive Night", price="€25"),
        ]))
        events = events_mod.find_events("2024-01-05", "2024-01-07", filters={"max_price": 15})
        assert len(events) == 1
        assert events[0]["name"] == "Cheap Night"

    def test_genre_filter_keeps_matching_only(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([
            _ra_row(title="Techno Night", genres=["Techno"]),
            _ra_row(title="House Night",  genres=["House"]),
        ]))
        events = events_mod.find_events("2024-01-05", "2024-01-07", filters={"genres": ["Techno"]})
        assert len(events) == 1
        assert events[0]["name"] == "Techno Night"

    def test_venue_filter_partial_case_insensitive_match(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([
            _ra_row(title="At Tresor",   venue="Tresor"),
            _ra_row(title="At Berghain", venue="Berghain"),
        ]))
        events = events_mod.find_events("2024-01-05", "2024-01-07", filters={"venue": "tresor"})
        assert len(events) == 1
        assert events[0]["name"] == "At Tresor"

    def test_area_filter_partial_match(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([
            _ra_row(title="In Mitte",     area="Mitte"),
            _ra_row(title="In Kreuzberg", area="Kreuzberg"),
        ]))
        events = events_mod.find_events("2024-01-05", "2024-01-07", filters={"area": "kreuzberg"})
        assert len(events) == 1
        assert events[0]["name"] == "In Kreuzberg"

    def test_http_error_returns_empty_list(self, mocker):
        import requests as req
        mocker.patch("requests.post", side_effect=req.RequestException("connection error"))
        events = events_mod.find_events("2024-01-05", "2024-01-07")
        assert events == []

    def test_graphql_errors_with_null_data_returns_empty(self, mocker):
        resp = Mock()
        resp.json.return_value = {"errors": [{"message": "field error"}], "data": None}
        resp.raise_for_status = Mock()
        mocker.patch("requests.post", return_value=resp)
        events = events_mod.find_events("2024-01-05", "2024-01-07")
        assert events == []

    def test_free_event_parsed_as_price_zero(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([_ra_row(price="Free")]))
        events = events_mod.find_events("2024-01-05", "2024-01-07")
        assert events[0]["price_numeric"] == 0.0

    def test_null_price_passes_max_price_filter(self, mocker):
        """Events with no listed price (price_numeric=None) must not be excluded by max_price."""
        mocker.patch("requests.post", return_value=_ra_response([_ra_row(price="")]))
        events = events_mod.find_events("2024-01-05", "2024-01-07", filters={"max_price": 15})
        assert len(events) == 1

    def test_content_url_prepended_with_ra_domain(self, mocker):
        mocker.patch("requests.post", return_value=_ra_response([_ra_row()]))
        events = events_mod.find_events("2024-01-05", "2024-01-07")
        assert events[0]["url"].startswith("https://ra.co")
