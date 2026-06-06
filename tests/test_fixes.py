"""
Tests for the round-3 fixes:

  - _annotate_price: currency is always shown, EUR included; Free / No price
  - weekend_dates: returns today + resolved weekend ISO dates
  - web_search: gap-honest, never raises, empty query is ungrounded
  - Mistral model: present in AVAILABLE_MODELS and routes to the mistral provider

All offline; web_search is exercised only on the no-network empty-query path,
and with ddgs mocked for the success path.
"""

from __future__ import annotations

import config
import llm_client
from prompts.system import weekend_dates
from tools import events as events_mod
from tools import web as web_mod


# ── _annotate_price ───────────────────────────────────────────────────────────

class TestAnnotatePrice:
    def test_eur_bare_number_gets_euro_symbol(self):
        # Berlin is seeded as Germany -> EUR; a bare RA price must get "€".
        assert events_mod._annotate_price("13/16", "Berlin") == "€13/16"

    def test_eur_already_symboled_is_unchanged(self):
        assert events_mod._annotate_price("€15", "Berlin") == "€15"

    def test_non_eur_city_gets_code_suffix(self, monkeypatch):
        # Inject a Copenhagen meta so currency resolves to DKK -> "kr".
        monkeypatch.setitem(
            events_mod._AREA_META_CACHE, "copenhagen", {"id": 1, "country": "Denmark"}
        )
        assert events_mod._annotate_price("150", "Copenhagen") == "150 kr"

    def test_empty_price_is_no_price_listed(self):
        assert events_mod._annotate_price("", "Berlin") == "No price listed"

    def test_free_is_free(self):
        assert events_mod._annotate_price("Free", "Berlin") == "Free"
        assert events_mod._annotate_price("free entry", "Berlin") == "Free"

    def test_non_numeric_passes_through(self):
        # No digits -> not a price; return as written (trimmed).
        assert events_mod._annotate_price("doors open", "Berlin") == "doors open"


# ── weekend_dates ─────────────────────────────────────────────────────────────

class TestWeekendDates:
    def test_returns_all_keys_in_iso(self):
        d = weekend_dates()
        for key in (
            "today", "this_friday", "this_saturday", "this_sunday",
            "next_friday", "next_saturday", "next_sunday",
        ):
            assert key in d, f"missing key {key}"
            assert len(d[key]) == 10 and d[key][4] == "-", f"{key} not ISO"

    def test_next_weekend_after_this_weekend(self):
        from datetime import date as _date
        d = weekend_dates()
        assert _date.fromisoformat(d["next_friday"]) > _date.fromisoformat(d["this_friday"])

    def test_this_friday_is_a_friday(self):
        from datetime import date as _date
        d = weekend_dates()
        assert _date.fromisoformat(d["this_friday"]).weekday() == 4


# ── web_search ────────────────────────────────────────────────────────────────

class TestWebSearch:
    def test_empty_query_is_ungrounded(self):
        out = web_mod.web_search("", k=3)
        assert out["grounded"] is False
        assert out["results"] == []

    def test_results_normalised_and_capped(self, mocker):
        class _FakeDDGS:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def text(self, q, max_results=5):
                return [
                    {"title": "Ben Klock news", "href": "https://x.example/1", "body": "A release."},
                    {"title": "More", "href": "https://x.example/2", "body": "Another."},
                ]
        import ddgs
        mocker.patch.object(ddgs, "DDGS", _FakeDDGS)
        out = web_mod.web_search("Ben Klock 2024", k=2)
        assert out["grounded"] is True
        assert len(out["results"]) == 2
        assert out["results"][0]["url"] == "https://x.example/1"
        assert out["results"][0]["title"] == "Ben Klock news"

    def test_ddgs_failure_is_gap_honest(self, mocker):
        class _BoomDDGS:
            def __enter__(self):
                raise RuntimeError("network down")
            def __exit__(self, *a):
                return False
        import ddgs
        mocker.patch.object(ddgs, "DDGS", _BoomDDGS)
        out = web_mod.web_search("anything", k=3)
        assert out["grounded"] is False
        assert out["results"] == []


# ── Mistral model wiring ──────────────────────────────────────────────────────

class TestMistralModel:
    def test_mistral_in_available_models(self):
        ids = [m["id"] for m in config.AVAILABLE_MODELS]
        assert "mistral-large-latest" in ids

    def test_mistral_routes_to_mistral_provider(self):
        assert llm_client._provider_for("mistral-large-latest") == "mistral"

    def test_haiku_still_default_and_openrouter(self):
        assert config.DEFAULT_MODEL == "anthropic/claude-haiku-4.5"
        assert llm_client._provider_for("anthropic/claude-haiku-4.5") == "openrouter"

    def test_mistral_has_price_entry(self):
        assert "mistral-large-latest" in config.MODEL_PRICES
