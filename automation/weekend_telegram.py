"""
Rave Atlas — weekend digest → Telegram (standalone, schedule-friendly).

WHY THIS IS SEPARATE FROM THE STREAMLIT APP
-------------------------------------------
Streamlit Community Cloud spins the app down when no browser is connected, so
the in-process APScheduler (automation/weekend_digest.py) can't be relied on to
fire on Friday morning. This module is a self-contained script designed to be
run by an external scheduler — a GitHub Actions cron (see
.github/workflows/weekend-digest.yml). It does NOT import Streamlit and does NOT
need the app to be awake.

WHAT IT DOES
------------
1. Computes the upcoming Fri → Tue window (Berlin's real weekend).
2. Fetches live Berlin events from Resident Advisor via find_events.
3. Formats a compact Telegram message — one line per event with the RA link to
   details/tickets — no LLM call, so there's nothing to fail or pay for.
4. Sends it via the Telegram Bot API.

It is Berlin-only by design: the weekend digest is the Berlin concierge feature.

CONFIG (env vars / GitHub Actions secrets)
------------------------------------------
    TELEGRAM_BOT_TOKEN   from @BotFather
    TELEGRAM_CHAT_ID     the chat/channel to post to (user, group, or @channel)

If either is missing the script logs and exits cleanly (no error), so the repo
stays runnable for contributors who haven't set Telegram up.
"""

from __future__ import annotations

import html
import sys
from datetime import date, timedelta

import requests

import config
from logging_config import get_logger
from tools.events import find_events

logger = get_logger(__name__)

_TELEGRAM_API = "https://api.telegram.org"
_HTTP_TIMEOUT = 15
_MAX_EVENTS = 12
_TELEGRAM_MAX_CHARS = 4096


# ── Date window ────────────────────────────────────────────────────────────────

def _weekend_window(today: date | None = None) -> tuple[date, date]:
    """Return (friday, tuesday) for the current/next weekend. Fri → Tue = 4 days."""
    today = today or date.today()
    days_ahead = (4 - today.weekday()) % 7  # 4 = Friday
    friday = today if days_ahead == 0 else today + timedelta(days=days_ahead)
    return friday, friday + timedelta(days=4)


# ── Message formatting (pure, testable) ────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(text or "", quote=False)


def format_digest_html(events: list[dict], date_from: str, date_to: str) -> str:
    """
    Build the Telegram message body (HTML parse mode).

    One line per event: a linked title (→ RA page), then venue · time · price.
    Pure function — no network — so it can be unit-tested without Telegram.
    """
    header = f"🎛️ <b>Berlin this weekend</b>  ({_esc(date_from)} → {_esc(date_to)})"

    if not events:
        return (
            f"{header}\n\n"
            "No Resident Advisor listings came back for this window. "
            'Check <a href="https://ra.co/events/de/berlin">ra.co/berlin</a> directly.'
        )

    lines = [header, ""]
    for evt in events[:_MAX_EVENTS]:
        name = _esc(evt.get("name") or "Untitled event")
        url = evt.get("url") or ""
        venue = _esc(evt.get("venue") or "")
        when = _esc(evt.get("start_time") or evt.get("date") or "")
        price = _esc(evt.get("price") or "")

        title = f'<a href="{html.escape(url, quote=True)}">{name}</a>' if url else f"<b>{name}</b>"
        meta_bits = [b for b in (venue, when, price) if b]
        meta = "  ·  ".join(meta_bits)
        lines.append(f"• {title}")
        if meta:
            lines.append(f"  <i>{meta}</i>")

    lines.append("")
    lines.append('🎟 Tap any title for details &amp; tickets on Resident Advisor.')

    msg = "\n".join(lines)
    if len(msg) > _TELEGRAM_MAX_CHARS:
        msg = msg[: _TELEGRAM_MAX_CHARS - 1].rstrip() + "…"
    return msg


# ── Telegram send ───────────────────────────────────────────────────────────────

def send_telegram_message(text: str) -> bool:
    """
    POST a message to the configured Telegram chat. Returns True on success.

    No-ops (returns False, logs) when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are
    not configured — so this is safe to run anywhere.
    """
    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        logger.warning("telegram_not_configured", has_token=bool(token), has_chat=bool(chat_id))
        return False

    try:
        resp = requests.post(
            f"{_TELEGRAM_API}/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("telegram_send_failed", error=str(exc))
        return False

    logger.info("telegram_sent", chars=len(text))
    return True


# ── Orchestration ────────────────────────────────────────────────────────────────

def run() -> bool:
    """Fetch Berlin weekend events, format, and send to Telegram. Returns success."""
    friday, tuesday = _weekend_window()
    date_from, date_to = friday.isoformat(), tuesday.isoformat()

    logger.info("telegram_digest_start", date_from=date_from, date_to=date_to)

    try:
        events = find_events(date_from, date_to)  # Berlin (HOME_CITY default)
    except Exception as exc:  # never let a fetch error crash the scheduled job
        logger.warning("telegram_find_events_failed", error=str(exc))
        events = []

    message = format_digest_html(events, date_from, date_to)
    ok = send_telegram_message(message)
    logger.info("telegram_digest_done", n_events=len(events), sent=ok)
    return ok


# ── CLI entry point (used by GitHub Actions) ─────────────────────────────────────

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    success = run()
    # Exit 0 even when Telegram is unconfigured — a missing optional integration
    # is not a CI failure. Only a hard send error after a valid attempt is non-zero.
    if not success and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        print("Telegram send failed despite configured credentials.", file=sys.stderr)
        sys.exit(1)
    print("Weekend Telegram digest job finished.")
