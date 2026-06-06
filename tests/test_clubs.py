"""
Rave Atlas, club registry, find_club tool, and club_scraper tests.

All offline: the scraper's network fetch and LLM extraction are mocked, so
these run deterministically in CI with no API keys and no HTTP.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_clubs_have_a_link(self):
        from tools.club_registry import ALL_CLUBS
        for c in ALL_CLUBS:
            assert c.website or c.berlin_de, f"{c.name} has no link at all"

    def test_no_duplicate_names(self):
        from tools.club_registry import ALL_CLUBS
        names = [c.name.lower() for c in ALL_CLUBS]
        assert len(names) == len(set(names)), "duplicate club names in registry"

    def test_scrape_values_are_valid(self):
        from tools.club_registry import ALL_CLUBS
        valid = {"http", "browser", "ra", "unknown"}
        for c in ALL_CLUBS:
            assert c.scrape in valid, f"{c.name} has invalid scrape={c.scrape!r}"

    def test_http_clubs_have_events_url(self):
        """Every http-scrapable club must have a URL for the scraper to hit."""
        from tools.club_registry import clubs_by_scrape_method
        for c in clubs_by_scrape_method("http"):
            assert c.events_url, f"{c.name} is scrape=http but has no events_url"

    def test_core_venues_present(self):
        from tools.club_registry import get_club
        for name in ("Berghain", "Tresor", "Sisyphos", "Wilde Renate", "OHM"):
            assert get_club(name) is not None, f"{name} missing from registry"


# ---------------------------------------------------------------------------
# find_club tool
# ---------------------------------------------------------------------------

class TestFindClub:
    def test_exact_hit(self):
        from tools.clubs import find_club
        r = find_club("Tresor")
        assert r["found"] is True
        assert r["website"] == "https://tresorberlin.com"
        assert "10179" in r["address"]

    def test_case_insensitive(self):
        from tools.clubs import find_club
        assert find_club("berghain")["found"] is True
        assert find_club("BERGHAIN")["found"] is True

    def test_fuzzy_partial(self):
        from tools.clubs import find_club
        r = find_club("visionaere")
        assert r["found"] is True
        assert "clubdervisionaere" in (r["website"] or "")

    def test_miss_returns_unfound(self):
        from tools.clubs import find_club
        r = find_club("Fabric London")
        assert r["found"] is False
        assert r["website"] is None
        assert r["events_url"] is None

    def test_link_only_club_resolves(self):
        from tools.clubs import find_club
        r = find_club("Void Club")
        assert r["found"] is True
        assert r["berlin_de"] # link-only clubs still carry the berlin.de page

    def test_return_shape(self):
        from tools.clubs import find_club
        r = find_club("Tresor")
        assert set(r.keys()) == {
            "found", "name", "address", "website", "events_url",
            "berlin_de", "instagram", "note", "suggestions",
        }


# ---------------------------------------------------------------------------
# Scraper: HTML -> text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_scripts_and_tags(self):
        from automation.club_scraper import html_to_text
        html = (
            "<html><head><style>.x{color:red}</style>"
            "<script>alert('x')</script></head>"
            "<body><h1>Events</h1><p>Klubnacht 6 June</p></body></html>"
        )
        text = html_to_text(html)
        assert "alert" not in text
        assert "color:red" not in text
        assert "Klubnacht 6 June" in text
        assert "Events" in text

    def test_decodes_entities(self):
        from automation.club_scraper import html_to_text
        assert "Beth & Co" in html_to_text("<p>Beth &amp; Co</p>")


# ---------------------------------------------------------------------------
# Scraper: JSON parsing
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_plain_json(self):
        from automation.club_scraper import _parse_json
        assert _parse_json('{"events": []}') == {"events": []}

    def test_code_fenced_json(self):
        from automation.club_scraper import _parse_json
        raw = '```json\n{"events": [{"name": "X"}]}\n```'
        assert _parse_json(raw)["events"][0]["name"] == "X"

    def test_garbage_returns_empty(self):
        from automation.club_scraper import _parse_json
        assert _parse_json("not json at all") == {"events": []}


# ---------------------------------------------------------------------------
# Scraper: LLM extraction (mocked llm_client.chat)
# ---------------------------------------------------------------------------

class TestExtractEvents:
    def test_extracts_and_cleans(self, monkeypatch):
        import automation.club_scraper as scraper

        fake = {
            "text": json.dumps({"events": [
                {"date": "2026-06-06", "name": "Klubnacht",
                 "lineup": "Steffi, Barker", "info": "Saturday, 19 euro"},
                {"date": "", "name": "", "lineup": "x", "info": "y"}, # dropped: no name
            ]})
        }
        monkeypatch.setattr(scraper.llm_client, "chat", lambda **kw: fake)
        events = scraper.extract_events("Berghain", "some page text")
        assert len(events) == 1
        assert events[0]["name"] == "Klubnacht"
        assert events[0]["date"] == "2026-06-06"

    def test_empty_text_skips_llm(self, monkeypatch):
        import automation.club_scraper as scraper

        def _boom(**kw):
            raise AssertionError("LLM should not be called for empty text")
        monkeypatch.setattr(scraper.llm_client, "chat", _boom)
        assert scraper.extract_events("X", " ") == []

    def test_llm_failure_returns_empty(self, monkeypatch):
        import automation.club_scraper as scraper

        def _raise(**kw):
            raise RuntimeError("provider down")
        monkeypatch.setattr(scraper.llm_client, "chat", _raise)
        assert scraper.extract_events("X", "real text") == []


# ---------------------------------------------------------------------------
# Scraper: orchestration (mocked fetch + extract), dry-run writes nothing
# ---------------------------------------------------------------------------

class TestScrapeAll:
    def test_dry_run_does_not_write(self, monkeypatch):
        import automation.club_scraper as scraper

        monkeypatch.setattr(scraper, "fetch", lambda url: "<p>Klubnacht</p>")
        monkeypatch.setattr(scraper, "fetch_browser", lambda url: "<p>Klubnacht</p>")
        monkeypatch.setattr(
            scraper, "extract_events",
            lambda name, text, model=None: [
                {"date": "2026-06-06", "name": "Klubnacht", "lineup": "", "info": ""}
            ],
        )

        def _no_write(*a, **k):
            raise AssertionError("dry-run must not write to SQLite")
        monkeypatch.setattr(scraper, "write_events", _no_write)

        summary = scraper.scrape_all(dry_run=True, limit=2)
        assert len(summary) == 2
        assert all(v == 1 for v in summary.values())

    def test_fetch_failure_marked_negative(self, monkeypatch):
        import automation.club_scraper as scraper

        monkeypatch.setattr(scraper, "fetch", lambda url: None) # all fetches fail
        monkeypatch.setattr(scraper, "fetch_browser", lambda url: None)
        summary = scraper.scrape_all(dry_run=True, limit=2)
        assert all(v == -1 for v in summary.values())
