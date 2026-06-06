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
import sqlite3
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

def _digest_window(today: date | None = None) -> tuple[date, date, str]:
    """
    Return (date_from, date_to, label) based on the actual day of the week.

    This makes the digest useful whenever it is triggered, not just on Fridays:

    - Mon/Tue/Wed: today through Sunday — "what is still on this week"
    - Thu: tomorrow (Fri) through Tuesday — "plan ahead for the weekend"
    - Fri/Sat/Sun: the Friday that opened this weekend through Tuesday —
      "it is happening right now / still going"

    The old _weekend_window() always looked for the NEXT Friday, so firing on
    Saturday showed next weekend instead of the current one. This fixes that.
    """
    today = today or date.today()
    wd = today.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    if wd <= 2:  # Mon, Tue, Wed
        date_from = today
        date_to = today + timedelta(days=6 - wd)  # through Sunday
        label = f"this week ({date_from.strftime('%d %b')} to {date_to.strftime('%d %b %Y')})"
    elif wd == 3:  # Thu
        date_from = today + timedelta(days=1)   # Friday
        date_to = date_from + timedelta(days=4)  # Tuesday
        label = f"this weekend ({date_from.strftime('%d %b')} to {date_to.strftime('%d %b %Y')})"
    else:  # Fri=4, Sat=5, Sun=6
        date_from = today - timedelta(days=wd - 4)  # anchor to this Friday
        date_to = date_from + timedelta(days=4)      # through Tuesday
        label = f"this weekend ({date_from.strftime('%d %b')} to {date_to.strftime('%d %b %Y')})"

    return date_from, date_to, label


# ── Scraped events (official club websites) ────────────────────────────────────

def _load_scraped_events(date_from: str, date_to: str) -> list[dict]:
    """
    Load events scraped from official club websites (club_events SQLite table).

    These come from automation/club_scraper.py — venues not on Resident Advisor
    like Renate, Heideglühen, Fitzroy, and any club whose events_url was scraped.
    Returns an empty list if the table doesn't exist yet (scraper hasn't run).

    The returned dicts use the same shape as RA events so format_digest_html
    can handle both uniformly. source="official" distinguishes them from RA.
    """
    try:
        conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        rows = conn.execute(
            """
            SELECT club, website, date, name, lineup, info
            FROM club_events
            WHERE (date = '' OR (date >= ? AND date <= ?))
            ORDER BY date, club
            """,
            (date_from, date_to),
        ).fetchall()
        conn.close()
        return [
            {
                "name": r[3],
                "venue": r[0],
                "url": r[1] or "",   # official website link, not RA
                "start_time": r[2],
                "lineup": r[4] or "",
                "price": r[5] or "",
                "source": "official",
            }
            for r in rows
            if r[3]  # skip empty names
        ]
    except Exception as exc:
        logger.warning("scraped_events_load_failed", error=str(exc)[:120])
        return []


# ── Message formatting (pure, testable) ────────────────────────────────────────

def _esc(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return html.escape(text or "", quote=False)


def format_digest_html(
    ra_events: list[dict],
    scraped_events: list[dict],
    date_from: str,
    date_to: str,
    label: str = "",
) -> str:
    """
    Build the Telegram message body (HTML parse mode).

    Merges RA events and scraped official-site events. RA events come first
    (live data, ticket links). Scraped events follow under a second heading
    when present (venues not on RA). Both sections link to their source.
    """
    header_label = label or f"{date_from} to {date_to}"
    header = f"🎛️ <b>Berlin {header_label}</b>"

    lines = [header, ""]

    if not ra_events and not scraped_events:
        lines.append(
            "No listings found for this window. "
            'Check <a href="https://ra.co/events/de/berlin">ra.co/berlin</a> directly.'
        )
        return "\n".join(lines)

    # ── RA events ──
    if ra_events:
        lines.append("<b>On Resident Advisor</b>")
        for evt in ra_events[:_MAX_EVENTS]:
            name = _esc(evt.get("name") or "Untitled event")
            url = evt.get("url") or ""
            venue = _esc(evt.get("venue") or "")
            when = _esc(evt.get("start_time") or evt.get("date") or "")
            price = _esc(evt.get("price") or "")
            title = f'<a href="{html.escape(url, quote=True)}">{name}</a>' if url else f"<b>{name}</b>"
            meta_bits = [b for b in (venue, when, price) if b]
            lines.append(f"• {title}")
            if meta_bits:
                lines.append(f"  <i>{'  ·  '.join(meta_bits)}</i>")

    # ── Official site events (not on RA) ──
    if scraped_events:
        lines.append("")
        lines.append("<b>Not on RA — official sites</b>")
        for evt in scraped_events[:8]:
            name = _esc(evt.get("name") or "Untitled event")
            url = evt.get("url") or ""
            venue = _esc(evt.get("venue") or "")
            when = _esc(evt.get("start_time") or "")
            lineup = _esc((evt.get("lineup") or "")[:80])
            title = f'<a href="{html.escape(url, quote=True)}">{name}</a>' if url else f"<b>{name}</b>"
            meta_bits = [b for b in (venue, when) if b]
            lines.append(f"• {title}")
            if meta_bits:
                lines.append(f"  <i>{'  ·  '.join(meta_bits)}</i>")
            if lineup:
                lines.append(f"  <i>{lineup}</i>")

    lines.append("")
    lines.append('🎟 Tap titles for details. RA links go to tickets; others to the club site.')
    lines.append(f'<i>window: {date_from} to {date_to}</i>')

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
    """
    Fetch Berlin events from all sources, format, and send to Telegram.

    Sources:
    1. Resident Advisor (GraphQL, live) — always attempted.
    2. Official club websites (SQLite club_events table) — included when the
       club scraper has been run in the same job or previously. Empty list if
       the table doesn't exist yet.

    Date window adapts to today's weekday so manual triggers are always
    relevant, not stuck pointing at the next Friday.
    """
    print(f"[weekend_telegram] v2 running from {__file__}", flush=True)
    d_from, d_to, label = _digest_window()
    date_from, date_to = d_from.isoformat(), d_to.isoformat()

    logger.info(
        "telegram_digest_start",
        date_from=date_from,
        date_to=date_to,
        label=label,
    )

    try:
        ra_events = find_events(date_from, date_to)
    except Exception as exc:
        logger.warning("telegram_find_events_failed", error=str(exc))
        ra_events = []

    scraped_events = _load_scraped_events(date_from, date_to)
    logger.info(
        "telegram_sources",
        ra=len(ra_events),
        scraped=len(scraped_events),
    )

    message = format_digest_html(ra_events, scraped_events, date_from, date_to, label)
    ok = send_telegram_message(message)
    logger.info(
        "telegram_digest_done",
        n_ra=len(ra_events),
        n_scraped=len(scraped_events),
        sent=ok,
    )
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
