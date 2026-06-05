"""
Tests for automation/weekend_telegram.py — the standalone Telegram digest sender.

Strategy:
  - format_digest_html: pure function — assert structure, linking, HTML escaping,
    empty-state handling, and the 4096-char Telegram cap. No network.
  - send_telegram_message: mock requests.post; verify the no-op-when-unconfigured
    path and the configured success path.
  - _weekend_window: deterministic Fri → Tue window.
"""
from __future__ import annotations

from datetime import date

import config
import automation.weekend_telegram as tg


# ── helpers ───────────────────────────────────────────────────────────────────

def _evt(name="Klock Night", url="https://ra.co/events/1", venue="Tresor",
         start_time="23:00", price="€18", genres=None) -> dict:
    return {
        "name": name, "url": url, "venue": venue, "start_time": start_time,
        "date": "2026-06-12", "price": price, "genres": genres or ["Techno"],
        "lineup": ["Ben Klock"], "area": "Mitte",
    }


# ── _weekend_window ─────────────────────────────────────────────────────────────

class TestWeekendWindow:
    def test_returns_friday_to_tuesday(self):
        # Wednesday 2026-06-10 → upcoming Friday 2026-06-12 → Tuesday 2026-06-16
        fri, tue = tg._weekend_window(today=date(2026, 6, 10))
        assert fri == date(2026, 6, 12)
        assert tue == date(2026, 6, 16)
        assert fri.weekday() == 4          # Friday
        assert (tue - fri).days == 4       # Fri → Tue

    def test_on_friday_uses_today(self):
        fri, tue = tg._weekend_window(today=date(2026, 6, 12))  # a Friday
        assert fri == date(2026, 6, 12)


# ── format_digest_html ───────────────────────────────────────────────────────

class TestFormatDigest:
    def test_links_each_event_to_its_ra_url(self):
        msg = tg.format_digest_html([_evt()], "2026-06-12", "2026-06-16")
        assert '<a href="https://ra.co/events/1">' in msg
        assert "Klock Night" in msg
        assert "Tresor" in msg
        assert "Berlin this weekend" in msg

    def test_empty_events_has_no_events_note_and_fallback_link(self):
        msg = tg.format_digest_html([], "2026-06-12", "2026-06-16")
        assert "No Resident Advisor listings" in msg
        assert "ra.co/events/de/berlin" in msg

    def test_html_special_chars_escaped(self):
        msg = tg.format_digest_html(
            [_evt(name="Bass & <Friends>", url="https://ra.co/events/2")],
            "2026-06-12", "2026-06-16",
        )
        assert "Bass &amp; &lt;Friends&gt;" in msg
        # The real URL still renders inside the href
        assert 'href="https://ra.co/events/2"' in msg

    def test_event_without_url_falls_back_to_bold(self):
        msg = tg.format_digest_html([_evt(url="")], "2026-06-12", "2026-06-16")
        assert "<b>Klock Night</b>" in msg
        assert "<a href" not in msg.split("Klock Night")[0][-30:]  # no link wrapping the name

    def test_truncates_to_telegram_limit(self):
        many = [_evt(name=f"Event number {i} with a long descriptive title", url=f"https://ra.co/events/{i}")
                for i in range(200)]
        msg = tg.format_digest_html(many, "2026-06-12", "2026-06-16")
        assert len(msg) <= tg._TELEGRAM_MAX_CHARS

    def test_caps_event_count(self):
        many = [_evt(name=f"E{i}", url=f"https://ra.co/events/{i}") for i in range(50)]
        msg = tg.format_digest_html(many, "2026-06-12", "2026-06-16")
        # Only the first _MAX_EVENTS titles should appear
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
        # Correct endpoint + payload shape
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
