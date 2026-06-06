"""
Tests for automation/weekend_telegram.py, the standalone Telegram digest sender.

Strategy:
  - _digest_window: deterministic window + label that adapts to the weekday.
  - _write_briefing: mock llm_client.chat; verify the model fallback chain and
    graceful empty return when every model fails.
  - format_digest_html: pure function, assert reason-first structure, linking,
    HTML escaping, empty-state handling, and the 4096-char Telegram cap.
  - send_telegram_message: mock requests.post; verify the no-op-when-unconfigured
    path and the configured success path.
No network in any test.
"""
from __future__ import annotations

from datetime import date

import config
import automation.weekend_telegram as tg


# ── helpers ───────────────────────────────────────────────────────────────────

def _evt(name="Klock Night", url="https://ra.co/events/1", venue="Tresor",
         price="€18", genres=None) -> dict:
    return {
        "name": name, "url": url, "venue": venue,
        "date": "2026-06-12", "date_label": "Fri 12 Jun",
        "start_time": "23:00", "end_time": "06:00", "time_label": "23:00 to 06:00",
        "price": price, "genres": genres or ["Techno"],
        "lineup": ["Ben Klock"], "area": "Mitte",
    }


# ── _digest_window ──────────────────────────────────────────────────────────────

class TestDigestWindow:
    def test_thursday_previews_the_weekend(self):
        # Thursday 2026-06-11 -> upcoming Friday through Tuesday.
        d_from, d_to, label = tg._digest_window(today=date(2026, 6, 11))
        assert d_from == date(2026, 6, 12)
        assert d_to == date(2026, 6, 16)
        assert d_from.weekday() == 4 # Friday
        assert (d_to - d_from).days == 4 # Fri to Tue
        assert "weekend" in label.lower()

    def test_friday_anchors_to_today(self):
        d_from, d_to, label = tg._digest_window(today=date(2026, 6, 12)) # a Friday
        assert d_from == date(2026, 6, 12)
        assert d_to == date(2026, 6, 16)

    def test_saturday_still_shows_this_weekend(self):
        # Saturday should anchor back to Friday, not jump to next weekend.
        d_from, _d_to, _label = tg._digest_window(today=date(2026, 6, 13))
        assert d_from == date(2026, 6, 12)

    def test_midweek_runs_today_through_sunday(self):
        # Monday 2026-06-08 -> today through Sunday 2026-06-14.
        d_from, d_to, label = tg._digest_window(today=date(2026, 6, 8))
        assert d_from == date(2026, 6, 8)
        assert d_to == date(2026, 6, 14)
        assert "week" in label.lower()


# ── _write_briefing (model fallback chain) ──────────────────────────────────────

class TestWriteBriefing:
    def test_empty_events_skips_the_model(self, mocker):
        chat = mocker.patch("llm_client.chat")
        assert tg._write_briefing([], "this weekend") == ""
        chat.assert_not_called()

    def test_returns_first_model_text(self, mocker):
        chat = mocker.patch("llm_client.chat", return_value={"text": "Quiet weekend, two real picks."})
        out = tg._write_briefing([_evt()], "this weekend")
        assert out == "Quiet weekend, two real picks."
        # First model in the chain is tried first.
        assert chat.call_args.kwargs["model"] == tg._BRIEFING_MODELS[0]

    def test_falls_back_to_second_model(self, mocker):
        def _side_effect(*_a, **kw):
            if kw["model"] == tg._BRIEFING_MODELS[0]:
                raise RuntimeError("mistral down")
            return {"text": "Haiku wrote this."}
        mocker.patch("llm_client.chat", side_effect=_side_effect)
        assert tg._write_briefing([_evt()], "this weekend") == "Haiku wrote this."

    def test_all_models_failing_returns_empty(self, mocker):
        mocker.patch("llm_client.chat", side_effect=RuntimeError("all down"))
        assert tg._write_briefing([_evt()], "this weekend") == ""


# ── format_digest_html ───────────────────────────────────────────────────────

class TestFormatDigest:
    def test_briefing_leads_then_lineup_links(self):
        msg = tg.format_digest_html(
            [_evt()], [], "2026-06-12", "2026-06-16",
            label="this weekend (12 Jun to 16 Jun 2026)",
            briefing="Tresor runs a hard Klockworks night, go for the kick weight.",
        )
        # Reason-first: the briefing text appears before the lineup heading.
        assert "Tresor runs a hard Klockworks night" in msg
        assert msg.index("Klockworks") < msg.index("The lineup")
        assert '<a href="https://ra.co/events/1">' in msg
        assert "Klock Night" in msg
        assert "<b>Berlin, this weekend" in msg
        # The cleaned time, not a raw timestamp, shows in the meta line.
        assert "23:00 to 06:00" in msg

    def test_works_without_a_briefing(self):
        msg = tg.format_digest_html([_evt()], [], "2026-06-12", "2026-06-16", label="this weekend")
        assert "The lineup" in msg
        assert '<a href="https://ra.co/events/1">' in msg

    def test_empty_events_has_fallback_link(self):
        msg = tg.format_digest_html([], [], "2026-06-12", "2026-06-16", label="this weekend")
        assert "No listings came back" in msg
        assert "ra.co/events/de/berlin" in msg

    def test_html_special_chars_escaped(self):
        msg = tg.format_digest_html(
            [_evt(name="Bass & <Friends>", url="https://ra.co/events/2")],
            [], "2026-06-12", "2026-06-16", label="this weekend",
            briefing="Watch the <crowd> & the door.",
        )
        assert "Bass &amp; &lt;Friends&gt;" in msg
        assert "&lt;crowd&gt; &amp; the door" in msg
        assert 'href="https://ra.co/events/2"' in msg

    def test_event_without_url_falls_back_to_bold(self):
        msg = tg.format_digest_html([_evt(url="")], [], "2026-06-12", "2026-06-16", label="x")
        assert "<b>Klock Night</b>" in msg

    def test_truncates_to_telegram_limit(self):
        many = [_evt(name=f"Event number {i} with a long descriptive title",
                     url=f"https://ra.co/events/{i}") for i in range(200)]
        msg = tg.format_digest_html(many, [], "2026-06-12", "2026-06-16", label="x")
        assert len(msg) <= tg._TELEGRAM_MAX_CHARS

    def test_caps_event_count(self):
        many = [_evt(name=f"E{i}", url=f"https://ra.co/events/{i}") for i in range(50)]
        msg = tg.format_digest_html(many, [], "2026-06-12", "2026-06-16", label="x")
        # Only the first _MAX_EVENTS events should be linked.
        assert f"https://ra.co/events/{tg._MAX_EVENTS}" not in msg


# ── send_telegram_message ────────────────────────────────────────────────────

class TestSendTelegram:
    def test_noop_when_unconfigured(self, mocker):
        mocker.patch.object(config, "TELEGRAM_BOT_TOKEN", "")
        mocker.patch.object(config, "TELEGRAM_CHAT_ID", "")
        post = mocker.patch("requests.post")
        assert tg.send_telegram_message("hi") is False
        post.assert_not_called()

    def test_sends_when_configured(self, mocker):
        mocker.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok123")
        mocker.patch.object(config, "TELEGRAM_CHAT_ID", "555")
        resp = mocker.Mock()
        resp.raise_for_status = mocker.Mock()
        post = mocker.patch("requests.post", return_value=resp)
        assert tg.send_telegram_message("<b>hi</b>") is True
        args, kwargs = post.call_args
        assert "bottok123/sendMessage" in args[0]
        assert kwargs["json"]["chat_id"] == "555"
        assert kwargs["json"]["parse_mode"] == "HTML"

    def test_send_failure_returns_false(self, mocker):
        import requests as req
        mocker.patch.object(config, "TELEGRAM_BOT_TOKEN", "tok")
        mocker.patch.object(config, "TELEGRAM_CHAT_ID", "1")
        mocker.patch("requests.post", side_effect=req.RequestException("network"))
        assert tg.send_telegram_message("x") is False
