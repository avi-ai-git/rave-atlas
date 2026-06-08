"""
Rave Atlas, persistent memory.

Two distinct concerns live here:

1. LangGraph SqliteSaver checkpointer
   Persists conversation thread state across Streamlit reruns (which
   re-execute the full script on every user interaction). Each session
   gets its own thread keyed by session_id.

2. SQLite taste profile + digest store
   A user model separate from conversation history. The taste profile
   (preferred genres, loved artists, budget ceiling, etc.) accumulates
   from explicit preferences and from thumbs-up/down feedback on events.
   Digests are stored by the weekend automation job and surfaced in the UI.

Both use the same SQLite file (SQLITE_PATH from config). SqliteSaver
manages its own checkpoint tables; we manage taste_profiles and digests.

Public interface consumed by agent.py and app.py:
    get_checkpointer() → SqliteSaver (singleton)
    load_profile(session_id) → dict | None
    save_profile(session_id, profile) → None
    update_profile_from_feedback(...) → dict (updated profile)
    save_digest(session_id, text) → None
    load_digest(session_id) → str | None
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

import config
from logging_config import get_logger

logger = get_logger(__name__)

# ── Default profile shape ─────────────────────────────────────────────────────
# Matches the keys that prompts/system.py _format_taste_profile() reads.
_DEFAULT_PROFILE: dict[str, Any] = {
    "preferred_genres": [],
    "blocked_genres": [],
    "loved_artists": [],
    "blocked_artists": [],
    "budget_ceiling": None,
    "preferred_areas": [],
}

# ── Singleton checkpointer ────────────────────────────────────────────────────
_checkpointer: SqliteSaver | None = None
_checkpointer_conn: sqlite3.Connection | None = None


# ── SQLite helpers ────────────────────────────────────────────────────────────

def _open_db() -> sqlite3.Connection:
    """
    Open a SQLite connection to SQLITE_PATH with WAL journal mode.
    Ensures the taste_profiles and digests tables exist.
    Called per-operation; caller is responsible for closing.
    """
    db_path = config.SQLITE_PATH
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # WAL unsupported on some cloud/network filesystems; continue without it
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS taste_profiles (
            session_id TEXT PRIMARY KEY,
            preferred_genres TEXT NOT NULL DEFAULT '[]',
            blocked_genres TEXT NOT NULL DEFAULT '[]',
            loved_artists TEXT NOT NULL DEFAULT '[]',
            blocked_artists TEXT NOT NULL DEFAULT '[]',
            budget_ceiling REAL,
            preferred_areas TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            digest_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_digests_session
            ON digests (session_id, created_at DESC);
    """)
    conn.commit()


def _row_to_profile(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "preferred_genres": json.loads(row["preferred_genres"]),
        "blocked_genres": json.loads(row["blocked_genres"]),
        "loved_artists": json.loads(row["loved_artists"]),
        "blocked_artists": json.loads(row["blocked_artists"]),
        "budget_ceiling": row["budget_ceiling"],
        "preferred_areas": json.loads(row["preferred_areas"]),
    }


# ── Checkpointer (for agent.py) ───────────────────────────────────────────────

def get_checkpointer() -> SqliteSaver:
    """
    Return the singleton LangGraph SqliteSaver.

    SqliteSaver persists LangGraph conversation-thread state so that
    Streamlit reruns (which re-execute the whole script) can resume the
    agent's memory without starting fresh each time.

    Uses a module-level singleton so one connection and one set of
    checkpoint tables are shared across the process lifetime.
    """
    global _checkpointer, _checkpointer_conn
    if _checkpointer is None:
        db_path = config.SQLITE_PATH
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _checkpointer_conn = sqlite3.connect(
            db_path, check_same_thread=False
        )
        _checkpointer = SqliteSaver(_checkpointer_conn)
        _checkpointer.setup()
        logger.info("checkpointer_initialised", path=config.SQLITE_PATH)
    return _checkpointer


def clear_thread_checkpoint(thread_id: str) -> None:
    """
    Delete all LangGraph checkpoint data for a given thread.

    Called when a thread is in a broken state — typically a tool_use block
    with no matching tool_result written to the checkpointer (happens when
    an agent turn crashes mid-call and the partial state gets persisted).
    Clearing it lets the next turn start from a clean slate rather than
    failing forever on the same corrupt state.

    Both checkpoint tables are keyed by thread_id:
      checkpoints  — the serialised graph state per turn
      writes       — the individual channel writes within a turn
    """
    try:
        conn = _open_db()
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
        logger.info("thread_checkpoint_cleared", thread_id=thread_id)
    except Exception as exc:
        logger.warning(
            "thread_checkpoint_clear_failed",
            thread_id=thread_id,
            error=str(exc),
        )


# ── Taste profile ─────────────────────────────────────────────────────────────

def load_profile(session_id: str) -> dict[str, Any] | None:
    """
    Return the taste profile for a session, or None if no profile exists yet.

    A None return is the signal to ask the user one quick clarifying question
    (handled in the system prompt) rather than guessing their preferences.
    """
    conn = _open_db()
    try:
        row = conn.execute(
            "SELECT * FROM taste_profiles WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    profile = _row_to_profile(row)
    logger.debug("profile_loaded", session_id=session_id, profile=profile)
    return profile


def save_profile(session_id: str, profile: dict[str, Any]) -> None:
    """
    Persist the taste profile for a session (insert or full replace).

    Any key absent from profile is written as an empty list / None,
    matching _DEFAULT_PROFILE semantics.
    """
    conn = _open_db()
    try:
        with conn: # auto-commit / rollback transaction
            conn.execute(
                """
                INSERT INTO taste_profiles (
                    session_id, preferred_genres, blocked_genres,
                    loved_artists, blocked_artists, budget_ceiling,
                    preferred_areas, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT (session_id) DO UPDATE SET
                    preferred_genres = excluded.preferred_genres,
                    blocked_genres = excluded.blocked_genres,
                    loved_artists = excluded.loved_artists,
                    blocked_artists = excluded.blocked_artists,
                    budget_ceiling = excluded.budget_ceiling,
                    preferred_areas = excluded.preferred_areas,
                    updated_at = datetime('now')
                """,
                (
                    session_id,
                    json.dumps(profile.get("preferred_genres", [])),
                    json.dumps(profile.get("blocked_genres", [])),
                    json.dumps(profile.get("loved_artists", [])),
                    json.dumps(profile.get("blocked_artists", [])),
                    profile.get("budget_ceiling"),
                    json.dumps(profile.get("preferred_areas", [])),
                ),
            )
    finally:
        conn.close()
    logger.info("profile_saved", session_id=session_id)


def update_profile_from_feedback(
    session_id: str,
    event: dict[str, Any],
    liked: bool,
) -> dict[str, Any]:
    """
    Update the taste profile based on a thumbs-up or thumbs-down on an event.

    Called by app.py when the user rates an event recommendation.
    Returns the full updated profile after saving.

    Logic:
      liked=True, add event's genres to preferred, artists to loved.
                    Remove them from blocked lists if they ended up there.
      liked=False, add genres to blocked (unless already preferred),
                    add artists to blocked (unless already loved).

    Budget ceiling is not updated from event feedback; it is set explicitly
    via the UI or conversation.
    """
    profile = load_profile(session_id) or dict(_DEFAULT_PROFILE)
    # deep-copy lists so we can mutate safely
    profile = {k: list(v) if isinstance(v, list) else v for k, v in profile.items()}

    genres: list[str] = event.get("genres") or []
    artists: list[str] = event.get("lineup") or []

    if liked:
        for g in genres:
            if g not in profile["preferred_genres"]:
                profile["preferred_genres"].append(g)
            if g in profile["blocked_genres"]:
                profile["blocked_genres"].remove(g)
        for a in artists:
            if a not in profile["loved_artists"]:
                profile["loved_artists"].append(a)
            if a in profile["blocked_artists"]:
                profile["blocked_artists"].remove(a)
    else:
        for g in genres:
            if (
                g not in profile["preferred_genres"]
                and g not in profile["blocked_genres"]
            ):
                profile["blocked_genres"].append(g)
        for a in artists:
            if (
                a not in profile["loved_artists"]
                and a not in profile["blocked_artists"]
            ):
                profile["blocked_artists"].append(a)

    save_profile(session_id, profile)
    logger.info(
        "profile_updated_from_feedback",
        session_id=session_id,
        liked=liked,
        event_genres=genres,
        event_artists=artists[:3],
    )
    return profile


# ── Digest store ──────────────────────────────────────────────────────────────

def save_digest(session_id: str, digest_text: str) -> None:
    """
    Store a weekend digest for a session.

    Written by automation/weekend_digest.py on Friday mornings.
    Multiple digests per session are allowed; load_digest returns the
    most recent one.
    """
    conn = _open_db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO digests (session_id, digest_text) VALUES (?, ?)",
                (session_id, digest_text),
            )
    finally:
        conn.close()
    logger.info("digest_saved", session_id=session_id, length=len(digest_text))


def load_digest(session_id: str) -> str | None:
    """Return the most recent digest for a session, or None if none stored."""
    conn = _open_db()
    try:
        row = conn.execute(
            """
            SELECT digest_text FROM digests
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return row["digest_text"] if row else None


def load_digest_with_age(session_id: str) -> tuple[str | None, float | None]:
    """Return (digest_text, age_in_hours) for the most recent digest, or (None, None).

    age_in_hours is computed in SQLite via julianday so it stays accurate
    regardless of the Python process clock.  Callers use the age to decide
    whether to auto-regenerate without a second DB round-trip.
    """
    conn = _open_db()
    try:
        row = conn.execute(
            """
            SELECT digest_text,
                   (julianday('now') - julianday(created_at)) * 24.0 AS age_hours
            FROM digests
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row:
        return row["digest_text"], row["age_hours"]
    return None, None


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import os

    # Use a temp file so the test doesn't pollute real data
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    _orig_path = config.SQLITE_PATH
    config.SQLITE_PATH = tmp.name

    try:
        SESSION = "test-session-001"

        print("=" * 60)
        print("Test 1: load_profile on empty DB -> None")
        print("=" * 60)
        result = load_profile(SESSION)
        assert result is None, f"FAIL: expected None, got {result}"
        print(" OK, None returned for new session")

        print()
        print("=" * 60)
        print("Test 2: save_profile + load_profile round-trip")
        print("=" * 60)
        profile_in = {
            "preferred_genres": ["Techno", "Minimal Techno"],
            "blocked_genres": ["Trance"],
            "loved_artists": ["Ben Klock", "Marcel Dettmann"],
            "blocked_artists": [],
            "budget_ceiling": 20.0,
            "preferred_areas": ["Friedrichshain"],
        }
        save_profile(SESSION, profile_in)
        profile_out = load_profile(SESSION)
        assert profile_out is not None, "FAIL: profile must exist after save"
        assert profile_out["preferred_genres"] == ["Techno", "Minimal Techno"], \
            f"FAIL: preferred_genres mismatch: {profile_out['preferred_genres']}"
        assert profile_out["budget_ceiling"] == 20.0, \
            f"FAIL: budget_ceiling mismatch: {profile_out['budget_ceiling']}"
        assert profile_out["preferred_areas"] == ["Friedrichshain"], \
            f"FAIL: preferred_areas mismatch: {profile_out['preferred_areas']}"
        print(f" preferred_genres : {profile_out['preferred_genres']}")
        print(f" loved_artists : {profile_out['loved_artists']}")
        print(f" budget_ceiling : {profile_out['budget_ceiling']}")
        print(" Round-trip OK")

        print()
        print("=" * 60)
        print("Test 3: update_profile_from_feedback, liked=True")
        print("=" * 60)
        fake_event = {
            "genres": ["Techno", "Electro"],
            "lineup": ["DVS1", "Surgeon"],
        }
        updated = update_profile_from_feedback(SESSION, fake_event, liked=True)
        assert "Electro" in updated["preferred_genres"], \
            "FAIL: new genre from liked event should be added to preferred"
        assert "DVS1" in updated["loved_artists"], \
            "FAIL: artist from liked event should be added to loved_artists"
        print(f" preferred_genres : {updated['preferred_genres']}")
        print(f" loved_artists : {updated['loved_artists']}")
        print(" liked=True update OK")

        print()
        print("=" * 60)
        print("Test 4: update_profile_from_feedback, liked=False")
        print("=" * 60)
        disliked_event = {
            "genres": ["House"],
            "lineup": ["SomeHouseArtist"],
        }
        updated2 = update_profile_from_feedback(SESSION, disliked_event, liked=False)
        assert "House" in updated2["blocked_genres"], \
            "FAIL: genre from disliked event should be added to blocked_genres"
        assert "SomeHouseArtist" in updated2["blocked_artists"], \
            "FAIL: artist from disliked event should be added to blocked_artists"
        # Already-preferred genres must NOT be blocked
        dislike_techno = update_profile_from_feedback(
            SESSION,
            {"genres": ["Techno"], "lineup": []},
            liked=False,
        )
        assert "Techno" not in dislike_techno["blocked_genres"], \
            "FAIL: preferred genre should never be added to blocked_genres"
        print(f" blocked_genres : {updated2['blocked_genres']}")
        print(f" blocked_artists : {updated2['blocked_artists']}")
        print(" liked=False update OK, preferred genre protection OK")

        print()
        print("=" * 60)
        print("Test 5: save_digest + load_digest")
        print("=" * 60)
        assert load_digest(SESSION) is None, "FAIL: no digest should exist yet"
        save_digest(SESSION, "This weekend: Berghain with Klock, Tresor with Dettmann.")
        digest = load_digest(SESSION)
        assert digest == "This weekend: Berghain with Klock, Tresor with Dettmann.", \
            f"FAIL: digest mismatch: {digest!r}"
        print(f" digest : {digest}")
        print(" Digest round-trip OK")

        print()
        print("=" * 60)
        print("Test 6: get_checkpointer, returns SqliteSaver")
        print("=" * 60)
        cp = get_checkpointer()
        assert isinstance(cp, SqliteSaver), \
            f"FAIL: expected SqliteSaver, got {type(cp)}"
        cp2 = get_checkpointer()
        assert cp is cp2, "FAIL: get_checkpointer must return the same singleton"
        print(f" checkpointer type : {type(cp).__name__}")
        print(" Singleton OK")

    finally:
        config.SQLITE_PATH = _orig_path
        # Close the singleton checkpointer connection before deleting the
        # temp file, Windows locks open file handles.
        if _checkpointer_conn is not None:
            _checkpointer_conn.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass # WAL side-files may linger briefly on Windows, not a bug

    print()
    print("All assertions passed.")
