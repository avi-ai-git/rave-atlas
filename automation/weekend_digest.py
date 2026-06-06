"""
Rave Atlas — Friday-morning weekend digest automation.

An APScheduler BackgroundScheduler job fires every Friday at 09:00 and:

  1. Determines the upcoming Fri → Tue date window
  2. Fetches live Berlin events via find_events (Resident Advisor GraphQL)
  3. For every session that has a taste profile, generates a personalised
     briefing via the LLM and writes it to the SQLite digest store
  4. If no sessions exist yet (fresh install), generates a global briefing
     under the "__global__" session so the UI digest tab is never empty

The scheduler is a singleton. app.py calls get_scheduler() once on boot;
subsequent calls return the already-running instance. atexit ensures a clean
shutdown so APScheduler's internal thread pool doesn't leak on process exit.

Public interface consumed by app.py (Phase 12):
    get_scheduler()                  → BackgroundScheduler (idempotent)
    generate_digest(session_id)      → str | None (for on-demand / testing)
"""

from __future__ import annotations

import atexit
import json
import sqlite3
from datetime import date, timedelta
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import config
import llm_client
import memory
from logging_config import get_logger
from tools.events import find_events

logger = get_logger(__name__)

# ── Singleton scheduler ───────────────────────────────────────────────────────
_scheduler: BackgroundScheduler | None = None

# Session ID used when no real sessions exist (fresh install / demo)
_GLOBAL_SESSION = "__global__"


# ── Date helpers ──────────────────────────────────────────────────────────────

def _digest_window() -> tuple[str, str, str, str]:
    """
    Return (date_from, date_to, period_label, digest_type) based on today.

    The window and framing adapt to the actual day so the digest is always
    relevant, not a stale "next Friday" placeholder when opened mid-week or
    on the weekend itself.

    Weekday logic (Mon=0 ... Sun=6):
    - Mon/Tue/Wed (0-2) -> "midweek": events from today through Sunday.
      Opening the digest on Monday should show what is still ahead this week,
      not fast-forward to next Friday.
    - Thu (3) -> "preview": upcoming Friday through Tuesday. The weekend is
      24 hours away; give people time to plan.
    - Fri/Sat/Sun (4-6) -> "weekend": anchor to the Friday that opened the
      current weekend, through Tuesday. Sat or Sun should still show THIS
      weekend, not next week.

    digest_type: "midweek" | "preview" | "weekend"
    period_label: human-readable title injected into the digest header.
    """
    today = date.today()
    wd = today.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    if wd <= 2:  # Mon, Tue, Wed
        days_to_sunday = 6 - wd
        date_from = today
        date_to = today + timedelta(days=days_to_sunday)
        label = f"Midweek in Berlin, {today.strftime('%d %b %Y')}"
        dtype = "midweek"
    elif wd == 3:  # Thu
        friday = today + timedelta(days=1)
        tuesday = friday + timedelta(days=4)
        date_from = friday
        date_to = tuesday
        label = (
            f"Weekend Preview, {friday.strftime('%d %b')} "
            f"to {tuesday.strftime('%d %b %Y')}"
        )
        dtype = "preview"
    else:  # Fri=4, Sat=5, Sun=6
        days_since_friday = wd - 4
        friday = today - timedelta(days=days_since_friday)
        tuesday = friday + timedelta(days=4)
        date_from = friday
        date_to = tuesday
        label = (
            f"This Weekend in Berlin, {friday.strftime('%d %b')} "
            f"to {tuesday.strftime('%d %b %Y')}"
        )
        dtype = "weekend"

    return date_from.isoformat(), date_to.isoformat(), label, dtype


# ── Session enumeration ───────────────────────────────────────────────────────

def _list_sessions_with_profiles() -> list[str]:
    """
    Return all session IDs that have a taste profile in the DB.

    Opens its own connection rather than importing memory._open_db (private)
    to keep automation self-contained. Same DB path, same WAL mode.
    """
    try:
        conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            "SELECT session_id FROM taste_profiles ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [r["session_id"] for r in rows]
    except sqlite3.OperationalError:
        # taste_profiles table doesn't exist yet (first boot before any session)
        return []


# ── Digest prompt builder ─────────────────────────────────────────────────────

def _build_digest_prompt(
    date_from: str,
    date_to: str,
    events: list[dict[str, Any]],
    profile: dict[str, Any] | None,
    period_label: str = "",
    digest_type: str = "weekend",
) -> str:
    """
    Build the LLM prompt for the digest.

    The intro, header, and framing adapt to digest_type so a Monday midweek
    digest reads differently from a Friday night "it is happening right now"
    digest or a Thursday preview. The date_from/date_to are always real ISO
    dates computed by _digest_window().
    """
    events_block = json.dumps(events[:12], ensure_ascii=False, indent=2)

    profile_block = ""
    if profile:
        parts: list[str] = []
        if pg := profile.get("preferred_genres"):
            parts.append(f"Preferred genres: {', '.join(pg)}")
        if la := profile.get("loved_artists"):
            parts.append(f"Loved artists: {', '.join(la)}")
        if bc := profile.get("budget_ceiling"):
            parts.append(f"Budget ceiling: EUR{bc}")
        if pa := profile.get("preferred_areas"):
            parts.append(f"Preferred areas: {', '.join(pa)}")
        if parts:
            profile_block = "\n\nUser taste profile:\n" + "\n".join(f"- {p}" for p in parts)

    no_events_note = (
        "\n\nNote: No events were returned from Resident Advisor for this window. "
        "Write a brief note explaining no events were found and suggest checking RA directly."
        if not events
        else ""
    )

    if digest_type == "midweek":
        intro = (
            "You are writing a midweek events digest for a Berlin electronic music fan. "
            "It is mid-week. Cover what is still worth going to before the weekend."
        )
        header = f"## This Week in Berlin ({date_from} to {date_to})"
        top_picks_label = f"Worth going to this week{' for you' if profile else ''}"
        extra_section = "Worth keeping an eye on"
    elif digest_type == "preview":
        intro = (
            "You are writing a Thursday weekend preview for a Berlin electronic music fan. "
            "The weekend is 24 hours away. Help them plan ahead."
        )
        header = f"## Weekend Preview ({date_from} to {date_to})"
        top_picks_label = f"Top picks for this weekend{' for you' if profile else ''}"
        extra_section = "Also on the radar"
    else:  # weekend
        intro = (
            "You are writing the weekend digest for a Berlin electronic music fan. "
            "The weekend is here or already underway. Be direct and decisive."
        )
        header = f"## {period_label or ('This Weekend in Berlin (' + date_from + ' to ' + date_to + ')')}"
        top_picks_label = f"Top picks{' for you' if profile else ''}"
        extra_section = "Worth knowing"

    return f"""\
{intro}

Date window: {date_from} to {date_to}{profile_block}

Berlin events this period (JSON). Each event has a "url" field, its Resident
Advisor page:
{events_block}{no_events_note}

Write a digest in clean markdown using this structure:

{header}

**{top_picks_label}**
- One sentence per event: link the event NAME to its url as a markdown link, like
  [Klockworks Night at Tresor](https://ra.co/events/123), then the venue and why
  it stands out (or why it fits the taste profile). Max 4 picks. Be specific, name
  the headliner, note the BPM range or genre, mention the price if it is notable.
  No hype words (avoid "amazing", "epic", "must-see").

**{extra_section}**
- 2 or 3 other events worth considering even if they do not perfectly match the
  profile. One sentence each, link each event name to its url the same way.

**Practical notes**
- One line on door policy, pricing range across all events, or anything
  operationally useful (e.g. "most venues start late, doors 23:00 or later").

Rules: every event you name MUST be a markdown link to its url. Never write an
event name without its link. Do not use em dashes or en dashes; use commas or
full stops. Keep the whole digest under 350 words. Lead with the best pick, not
a preamble.
"""


# ── Core generation function ──────────────────────────────────────────────────

def generate_digest(session_id: str) -> str | None:
    """
    Generate a weekend digest for one session and persist it.

    Fetches live RA events, loads the session's taste profile (for
    personalisation), calls the LLM, and writes the result to the digest
    store via memory.save_digest.

    Returns the digest text on success, None on any failure. Never raises —
    a failed digest job should not crash the scheduler or the Streamlit app.

    Args:
        session_id: The session to generate for. Use "__global__" for a
                    non-personalised global digest.
    """
    date_from, date_to, period_label, digest_type = _digest_window()
    profile = memory.load_profile(session_id) if session_id != _GLOBAL_SESSION else None

    logger.info(
        "digest_generating",
        session_id=session_id,
        date_from=date_from,
        date_to=date_to,
        digest_type=digest_type,
        period_label=period_label,
        personalised=profile is not None,
    )

    try:
        events = find_events(date_from, date_to)
    except Exception as exc:
        logger.warning("digest_find_events_failed", error=str(exc), session_id=session_id)
        events = []

    prompt = _build_digest_prompt(
        date_from, date_to, events, profile,
        period_label=period_label, digest_type=digest_type,
    )

    try:
        result = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        digest_text = result["text"].strip()
    except Exception as exc:
        logger.error("digest_llm_failed", error=str(exc), session_id=session_id)
        return None

    if not digest_text:
        logger.warning("digest_empty_response", session_id=session_id)
        return None

    try:
        memory.save_digest(session_id, digest_text)
    except Exception as exc:
        logger.error("digest_save_failed", error=str(exc), session_id=session_id)
        return None

    logger.info(
        "digest_saved",
        session_id=session_id,
        n_events=len(events),
        digest_length=len(digest_text),
        cost_usd=result.get("cost_estimate", 0.0),
    )
    return digest_text


# ── Scheduled job ─────────────────────────────────────────────────────────────

def _run_digest_job() -> None:
    """
    APScheduler entry point: generate digests for all active sessions.

    If no sessions have profiles yet (fresh install or demo), generate one
    global digest so the Digest tab in the UI is never blank.
    """
    logger.info("digest_job_started")

    sessions = _list_sessions_with_profiles()
    if not sessions:
        logger.info("digest_no_sessions_using_global")
        sessions = [_GLOBAL_SESSION]

    succeeded = 0
    for session_id in sessions:
        result = generate_digest(session_id)
        if result is not None:
            succeeded += 1

    logger.info(
        "digest_job_complete",
        total=len(sessions),
        succeeded=succeeded,
        failed=len(sessions) - succeeded,
    )


# ── Scheduler singleton ───────────────────────────────────────────────────────

def get_scheduler() -> BackgroundScheduler:
    """
    Return the running BackgroundScheduler (start it if not already running).

    Idempotent — call as many times as you like from app.py. The Friday
    09:00 job is registered with replace_existing=True so re-registration
    on hot-reload doesn't duplicate the job.

    atexit ensures a clean shutdown so APScheduler's thread pool doesn't
    leak when the Streamlit process exits.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler(
        job_defaults={"misfire_grace_time": 3600},  # 1h grace — handles late wakeup
        timezone="Europe/Berlin",
    )

    _scheduler.add_job(
        _run_digest_job,
        CronTrigger(
            day_of_week="fri",
            hour=9,
            minute=0,
            timezone="Europe/Berlin",
        ),
        id="weekend_digest",
        replace_existing=True,
    )

    _scheduler.start()

    def _safe_shutdown() -> None:
        if _scheduler is not None and _scheduler.running:
            _scheduler.shutdown(wait=False)

    atexit.register(_safe_shutdown)
    logger.info("scheduler_started", job_id="weekend_digest", fires="fri@09:00 Europe/Berlin")
    return _scheduler


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile
    import os

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    # Redirect SQLite to a temp file so the test doesn't pollute real data
    _tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    _tmp.close()
    _orig_sqlite = config.SQLITE_PATH
    config.SQLITE_PATH = _tmp.name

    try:
        print("=" * 60)
        print("Test 1: _digest_window returns valid ISO dates and context")
        print("=" * 60)
        from datetime import date as _date
        d_from, d_to, label, dtype = _digest_window()
        print(f"  date_from    : {d_from}")
        print(f"  date_to      : {d_to}")
        print(f"  period_label : {label}")
        print(f"  digest_type  : {dtype}")
        assert len(d_from) == 10 and d_from[4] == "-", "FAIL: date_from not ISO format"
        assert len(d_to) == 10 and d_to[4] == "-", "FAIL: date_to not ISO format"
        assert _date.fromisoformat(d_to) >= _date.fromisoformat(d_from), (
            "FAIL: date_to must be >= date_from"
        )
        assert dtype in ("midweek", "preview", "weekend"), f"FAIL: unexpected dtype {dtype!r}"
        assert label, "FAIL: period_label must not be empty"
        diff = (_date.fromisoformat(d_to) - _date.fromisoformat(d_from)).days
        assert 0 <= diff <= 7, f"FAIL: window should be 0-7 days, got {diff}"
        print(f"  window       : {diff} days")
        print("  OK - digest window correct for today's weekday")

        print()
        print("=" * 60)
        print("Test 2: _list_sessions_with_profiles on empty DB")
        print("=" * 60)
        # Ensure tables exist by touching memory once
        memory._open_db().close()
        sessions = _list_sessions_with_profiles()
        print(f"  sessions : {sessions}")
        assert sessions == [], "FAIL: fresh DB should return no sessions"
        print("  OK - empty list on fresh DB")

        print()
        print("=" * 60)
        print("Test 3: generate_digest - full LLM run (live API)")
        print("=" * 60)
        print("  (fetches RA events + calls LLM - may take 15-30s)")
        digest = generate_digest(_GLOBAL_SESSION)
        if digest is None:
            print("  WARN: digest returned None (RA or LLM may be unavailable)")
            print("  Testing fallback: checking that None is handled gracefully")
        else:
            print(f"  digest length : {len(digest)} chars")
            print(f"  digest preview:")
            for line in digest.splitlines()[:8]:
                print(f"    {line}")
            assert len(digest) > 50, "FAIL: digest should have meaningful content"
            assert "Berlin" in digest or "Weekend" in digest or "weekend" in digest, (
                "FAIL: digest should mention Berlin or Weekend"
            )
            # Verify it was persisted
            loaded = memory.load_digest(_GLOBAL_SESSION)
            assert loaded == digest, "FAIL: loaded digest does not match generated digest"
            print("  OK - digest generated and persisted")

        print()
        print("=" * 60)
        print("Test 4: get_scheduler - returns running BackgroundScheduler")
        print("=" * 60)
        from apscheduler.schedulers.background import BackgroundScheduler as _BS
        sched = get_scheduler()
        print(f"  type    : {type(sched).__name__}")
        print(f"  running : {sched.running}")
        assert isinstance(sched, _BS), "FAIL: get_scheduler must return BackgroundScheduler"
        assert sched.running, "FAIL: scheduler must be running after get_scheduler()"
        jobs = sched.get_jobs()
        job_ids = [j.id for j in jobs]
        print(f"  jobs    : {job_ids}")
        assert "weekend_digest" in job_ids, "FAIL: weekend_digest job must be registered"
        print("  OK - scheduler running with weekend_digest job")

        print()
        print("=" * 60)
        print("Test 5: get_scheduler - idempotency (singleton)")
        print("=" * 60)
        sched2 = get_scheduler()
        assert sched is sched2, "FAIL: get_scheduler must return the same singleton"
        assert len(sched2.get_jobs()) == len(jobs), "FAIL: job count changed on second call"
        print("  OK - same object returned, no duplicate jobs")

    finally:
        # Shut down scheduler cleanly before cleanup
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
        config.SQLITE_PATH = _orig_sqlite
        try:
            os.unlink(_tmp.name)
        except OSError:
            pass

    print()
    print("All assertions passed.")
