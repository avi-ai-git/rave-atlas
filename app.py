"""
Rave Atlas — Streamlit UI (Phase 12).

Three tabs, one shared agent, one session ID per browser session:

  🗓 This Weekend  — event discovery, compare, and ratings
  📚 Learn         — music education chat (genres, history, labels, theory)
  🎛  Crate        — set-list builder with playable Deezer previews

Sidebar: model picker, tone selector, temperature + top-p sliders, session
cost/token display, clear-chat button.

All agent logic lives in agent.py. This file is pure Streamlit routing and
presentation — no LLM calls, no tool logic, no DB access beyond memory helpers.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st

import config
import memory
from agent import run_agent
from automation.weekend_digest import generate_digest, get_scheduler


# ── Knowledge-base warm-up (Streamlit Cloud cold start) ───────────────────────

@st.cache_resource(show_spinner="Building knowledge base — first run only…")
def _ensure_kb_seeded() -> None:
    """
    Seed ChromaDB on the first process start.

    On Streamlit Cloud the data/ directory is empty on every fresh container.
    This runs ingest() once (downloading the ~80 MB sentence-transformer model
    and embedding all seven KB files) so that explain_music works immediately.
    Cached by @st.cache_resource, so it runs at most once per server process.
    """
    from ingest import get_collection, ingest
    col = get_collection()
    if col.count() == 0:
        ingest()


# ── Session state initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    """Set up all session_state keys on first load."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    for tab in ("weekend", "learn", "crate"):
        st.session_state.setdefault(f"messages_{tab}", [])
    st.session_state.setdefault("total_cost", 0.0)
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("rated_events", set())


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict[str, Any]:
    """Render sidebar controls; return selected settings dict."""
    with st.sidebar:
        st.header("Settings")

        model_names = [m["name"] for m in config.AVAILABLE_MODELS]
        model_ids   = [m["id"]   for m in config.AVAILABLE_MODELS]
        default_idx = (
            model_ids.index(config.DEFAULT_MODEL)
            if config.DEFAULT_MODEL in model_ids else 0
        )
        selected_name = st.selectbox("Model", model_names, index=default_idx)
        model_id = model_ids[model_names.index(selected_name)]

        tone = st.radio(
            "Tone",
            options=["friendly", "concise", "formal"],
            index=0,
            horizontal=True,
        )

        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        top_p       = st.slider("Top-p",       0.5, 1.0, 1.0, 0.05)

        st.divider()

        st.caption("Session stats")
        col_tok, col_cost = st.columns(2)
        col_tok.metric("Tokens",  f"{st.session_state['total_tokens']:,}")
        col_cost.metric("Cost",   f"${st.session_state['total_cost']:.4f}")

        st.divider()

        if st.button("Clear chat history", use_container_width=True):
            for tab in ("weekend", "learn", "crate"):
                st.session_state[f"messages_{tab}"] = []
            st.session_state["total_cost"]    = 0.0
            st.session_state["total_tokens"]  = 0
            st.session_state["rated_events"]  = set()
            st.rerun()

        st.divider()
        short_id = st.session_state["session_id"][:8]
        st.caption(f"Session `{short_id}…`")

    return {
        "model_id":    model_id,
        "tone":        tone,
        "temperature": temperature,
        "top_p":       top_p,
    }


# ── Tool-trace expander ───────────────────────────────────────────────────────

def _render_tool_trace(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool trace — {len(tool_calls)} call(s)", expanded=False):
        for i, call in enumerate(tool_calls, 1):
            st.markdown(f"**{i}. `{call['name']}`**")
            if call.get("args"):
                st.json(call["args"], expanded=False)
            preview = call.get("output_preview", "")
            if preview:
                st.code(preview, language=None)
            if i < len(tool_calls):
                st.divider()


# ── Cost / token badge ────────────────────────────────────────────────────────

def _render_cost_badge(model: str, usage: dict, cost: float) -> None:
    st.caption(
        f"`{model}` · "
        f"{usage.get('total_tokens', 0):,} tokens "
        f"({usage.get('prompt_tokens', 0):,} in / "
        f"{usage.get('completion_tokens', 0):,} out) · "
        f"${cost:.5f}"
    )


# ── Event ratings ─────────────────────────────────────────────────────────────

def _render_event_ratings(tool_calls: list[dict], msg_idx: int) -> None:
    """
    Show thumbs-up / thumbs-down per ranked event when compare_events ran.
    A click updates the SQLite taste profile via memory.update_profile_from_feedback.
    """
    compare_call = next(
        (c for c in tool_calls if c["name"] == "compare_events"), None
    )
    if not compare_call or not compare_call.get("full_output"):
        return

    try:
        ranked_data = json.loads(compare_call["full_output"])
        ranked = ranked_data.get("ranked_events", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return

    if not ranked:
        return

    # Raw event dicts (for genres + lineup) come from the find_events call
    find_call = next(
        (c for c in tool_calls if c["name"] == "find_events"), None
    )
    raw_events: list[dict] = []
    if find_call and find_call.get("full_output"):
        try:
            raw_events = json.loads(find_call["full_output"])
        except (json.JSONDecodeError, TypeError):
            pass

    session_id = st.session_state["session_id"]
    rated      = st.session_state["rated_events"]

    st.markdown("---")
    st.markdown("**Rate these picks to improve future recommendations:**")

    for evt in ranked[:5]:
        name       = evt.get("event_name", "")
        fit        = evt.get("fit_summary", "")
        rank       = evt.get("rank", "?")
        rating_key = f"r_{msg_idx}_{name}"

        if rating_key in rated:
            st.caption(f"#{rank} {name} — rated ✓")
            continue

        col_label, col_up, col_down = st.columns([7, 1, 1])
        col_label.markdown(f"**#{rank} {name}** — {fit}")

        if col_up.button("👍", key=f"up_{msg_idx}_{name}", help="Good fit"):
            raw = next((e for e in raw_events if e.get("name") == name), {})
            memory.update_profile_from_feedback(session_id, raw, liked=True)
            rated.add(rating_key)
            st.toast(f"Liked — profile updated", icon="✅")
            st.rerun()

        if col_down.button("👎", key=f"dn_{msg_idx}_{name}", help="Not for me"):
            raw = next((e for e in raw_events if e.get("name") == name), {})
            memory.update_profile_from_feedback(session_id, raw, liked=False)
            rated.add(rating_key)
            st.toast("Noted — will avoid similar events", icon="🚫")
            st.rerun()


# ── Setlist card renderer ─────────────────────────────────────────────────────

def _render_setlist(tool_calls: list[dict]) -> None:
    """
    Render the structured set list — title, energy arc, per-track cards
    with Deezer audio player and YouTube link — when build_setlist ran.
    """
    sl_call = next(
        (c for c in tool_calls if c["name"] == "build_setlist"), None
    )
    if not sl_call or not sl_call.get("full_output"):
        return

    try:
        sl = json.loads(sl_call["full_output"])
    except (json.JSONDecodeError, TypeError):
        return

    tracks     = sl.get("tracks", [])
    energy_arc = sl.get("energy_arc", [])

    if not tracks:
        return

    st.markdown("---")
    st.markdown(f"### {sl.get('title', 'Set')}")

    if energy_arc:
        arc_str = " → ".join(str(e) for e in energy_arc)
        st.caption(f"Energy arc: **{arc_str}**")

    for i, track in enumerate(tracks, 1):
        energy      = track.get("energy", 5)
        preview_url = track.get("preview_url")
        deezer_url  = track.get("deezer_url")
        youtube_url = track.get("youtube_url", "")
        artist      = track.get("artist", "")
        title       = track.get("title", "")
        reason      = track.get("reason", "")

        with st.container(border=True):
            col_num, col_info, col_nrg = st.columns([1, 9, 2])
            col_num.markdown(f"**{i}**")
            col_info.markdown(f"**{artist}** — {title}")
            col_nrg.markdown(f"⚡ {energy}/10")

            if reason:
                st.caption(reason)

            link_parts: list[str] = []
            if deezer_url:
                link_parts.append(f"[Deezer]({deezer_url})")
            if youtube_url:
                link_parts.append(f"[YouTube ↗]({youtube_url})")
            if link_parts:
                st.caption(" · ".join(link_parts))

            if preview_url:
                st.audio(preview_url, format="audio/mp3")


# ── Shared message renderer ───────────────────────────────────────────────────

def _render_messages(tab_key: str) -> None:
    """Render all stored messages for a tab from session_state."""
    for idx, msg in enumerate(st.session_state[f"messages_{tab_key}"]):
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.warning(msg["content"])
            else:
                st.markdown(msg["content"])

            if msg["role"] == "assistant" and not msg.get("blocked"):
                tool_calls = msg.get("tool_calls", [])
                _render_tool_trace(tool_calls)
                if msg.get("usage"):
                    _render_cost_badge(
                        msg.get("model", ""),
                        msg["usage"],
                        msg.get("cost_estimate", 0.0),
                    )
                if tab_key == "weekend":
                    _render_event_ratings(tool_calls, idx)
                if tab_key == "crate":
                    _render_setlist(tool_calls)


# ── Chat input handler ────────────────────────────────────────────────────────

def _handle_chat_input(
    tab_key: str,
    settings: dict,
    placeholder: str,
) -> None:
    """
    Accept a chat_input submission, run the agent, and append the result
    to the tab's message list.

    Streamlit reruns the script on every interaction. On the run where
    chat_input fires, we render the new messages inline (so the user
    sees the response immediately). On subsequent reruns, _render_messages
    picks them up from session_state — no double-render because
    chat_input returns None after the first submit.
    """
    prompt = st.chat_input(placeholder)
    if not prompt:
        return

    session_id = st.session_state["session_id"]
    messages   = st.session_state[f"messages_{tab_key}"]

    # User turn
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent turn
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = run_agent(
                message=prompt,
                session_id=session_id,
                model_id=settings["model_id"],
                tone=settings["tone"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
            )

        if result["blocked"]:
            st.warning(result["text"])
        else:
            st.markdown(result["text"])
            tool_calls = result["tool_calls"]
            _render_tool_trace(tool_calls)
            _render_cost_badge(result["model"], result["usage"], result["cost_estimate"])
            # Inline render for this response (before next rerun)
            if tab_key == "weekend":
                _render_event_ratings(tool_calls, len(messages))
            if tab_key == "crate":
                _render_setlist(tool_calls)

    messages.append({
        "role":          "assistant",
        "content":       result["text"],
        "blocked":       result["blocked"],
        "tool_calls":    result["tool_calls"],
        "usage":         result["usage"],
        "cost_estimate": result["cost_estimate"],
        "model":         result["model"],
    })

    st.session_state["total_cost"]   += result["cost_estimate"]
    st.session_state["total_tokens"] += result["usage"]["total_tokens"]


# ── Digest section ────────────────────────────────────────────────────────────

def _render_digest_section() -> None:
    """
    Show the most recent weekend digest for this session (or the global one
    written by the scheduler). Offers an on-demand regenerate button so the
    user isn't dependent on the Friday 09:00 scheduler firing.
    """
    session_id = st.session_state["session_id"]
    digest = memory.load_digest(session_id) or memory.load_digest("__global__")

    with st.expander(
        "📋 This weekend's digest" + ("" if digest else " — not yet generated"),
        expanded=not digest,
    ):
        if digest:
            st.markdown(digest)
        else:
            st.info(
                "No digest yet. The scheduler runs every Friday at 09:00 Berlin time, "
                "or click below to generate one now."
            )

        if st.button("Generate / refresh digest", key="btn_regen_digest"):
            with st.spinner("Fetching events and writing digest…"):
                new_digest = generate_digest(session_id)
            if new_digest:
                st.success("Digest ready.")
                st.rerun()
            else:
                st.warning(
                    "Could not generate digest — Resident Advisor or the LLM "
                    "may be temporarily unavailable."
                )


# ── Tab content functions ─────────────────────────────────────────────────────

def _tab_weekend(settings: dict) -> None:
    st.subheader("Weekend concierge")
    st.caption(
        "Ask what's on this Friday, compare venues, or describe your taste "
        "and get personalised picks. Rate events to improve future suggestions."
    )
    _render_digest_section()
    st.divider()
    _render_messages("weekend")
    _handle_chat_input(
        "weekend", settings,
        placeholder="What's on this Friday? / Find techno under €20 / What's at Tresor tonight?",
    )


def _tab_learn(settings: dict) -> None:
    st.subheader("Learn the music")
    st.caption(
        "Ask about genres, BPM signatures, Berlin scene history, record labels, "
        "or how a track is structured. Answers come from the curated Rave Atlas knowledge base."
    )
    _render_messages("learn")
    _handle_chat_input(
        "learn", settings,
        placeholder="What defines minimal techno? / History of Tresor Records / How does a DJ read a crowd?",
    )


def _tab_crate(settings: dict) -> None:
    st.subheader("Crate — set builder")
    st.caption(
        "Describe the vibe and get a set with an energy arc, "
        "30-second Deezer previews, and YouTube links."
    )
    _render_messages("crate")
    _handle_chat_input(
        "crate", settings,
        placeholder="Build a hypnotic 2am techno set / Warm-up for Watergate Friday 23h",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Rave Atlas",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _ensure_kb_seeded()  # no-op after first run; seeds ChromaDB on cold start
    _init_session()
    get_scheduler()   # idempotent — starts the Friday digest job if not running

    settings = _render_sidebar()

    st.title("🎛️ Rave Atlas")
    st.caption(
        "Berlin's electronic music scene — weekend planner, music education, set-list builder."
    )

    tab_weekend, tab_learn, tab_crate = st.tabs(
        ["🗓 This Weekend", "📚 Learn", "🎛 Crate"]
    )

    with tab_weekend:
        _tab_weekend(settings)

    with tab_learn:
        _tab_learn(settings)

    with tab_crate:
        _tab_crate(settings)


if __name__ == "__main__":
    main()
