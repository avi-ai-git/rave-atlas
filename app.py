"""
Berlin Rave Atlas, Streamlit UI.

Four tabs, a Berlin-first agent, and two deliberate non-agent browsers.

  Your Berlin Guide      Unified agent chat: events, scene knowledge, artists (ReAct agent)
  Berlin Raves           AI picks digest + raw RA browse for Berlin (direct tool call)
  Rave Set Builder       1-hour set builder (direct, deterministic tool call)
  Raves Beyond Berlin    browse all European Resident Advisor cities (plain API browse)

Why the split. The agent earns its cost where runtime reasoning matters: picking
which night fits your taste, explaining a label's history, recommending what to
wear. Berlin Raves, Rave Set Builder, and Raves Beyond Berlin are each one
well-defined job, so they call the tool directly rather than asking the model
to decide to call it. Faster, cheaper, and no failure mode.

Chat. The guide keeps a single message list and a single agent thread so
event queries and music questions share context naturally. Messages render
oldest to newest directly in the page; Streamlit's chat_input is sticky at the
bottom. On submit we append and rerun, so everything renders through one path.

All agent and tool logic lives in agent.py and tools/. This file is presentation
and routing only, no LLM calls except the direct build_setlist used by the set
builder.
"""

from __future__ import annotations

# ChromaDB needs sqlite3 >= 3.35, but Streamlit Cloud's base image ships an
# older system sqlite3. Swap in the bundled pysqlite3 BEFORE anything imports
# chromadb (transitively via agent -> tools.music_kb -> ingest). This must be
# the first executable code after __future__. On local dev / Windows the
# pysqlite3-binary wheel isn't installed (platform marker in requirements), so
# the except branch keeps the stdlib sqlite3, which is already new enough.
import sys

try:
    import pysqlite3 # type: ignore

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

import json
import uuid
from datetime import date, timedelta
from typing import Any

import streamlit as st

import config
import llm_client
import memory
from agent import run_agent
from automation.weekend_digest import generate_digest, get_scheduler
from tools.setlist import build_setlist


# ── Theme polish (CSS) ────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Layout ──────────────────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 5rem; /* room for the sticky chat input */
    max-width: 860px;
}
h1 { letter-spacing: -0.03em; font-weight: 800; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid #333; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    color: #aaa;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ddd !important;
    background: rgba(255,255,255,0.04);
}
.stTabs [aria-selected="true"] {
    background: rgba(214,48,49,0.12);
    border-bottom: 2px solid #D63031;
    color: #F5F5F5 !important;
}

/* ── Chat messages ────────────────────────────────────────────────────────── */
.stChatMessage {
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.5rem;
}
/* User bubble: slightly red-tinted dark */
[data-testid="stChatMessage"][data-role="user"] {
    background: rgba(214,48,49,0.08) !important;
    border: 1px solid rgba(214,48,49,0.2);
}
/* Assistant bubble: neutral dark */
[data-testid="stChatMessage"][data-role="assistant"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Chat input: visually distinct from message area ─────────────────────── */
.stChatInput > div {
    border: 2px solid rgba(214,48,49,0.45) !important;
    border-radius: 14px !important;
    background: #111418 !important;
}
.stChatInput > div:focus-within {
    border-color: #D63031 !important;
    box-shadow: 0 0 0 2px rgba(214,48,49,0.2) !important;
}
.stChatInput textarea {
    border-radius: 12px !important;
    background: transparent !important;
    font-size: 0.97rem;
    min-height: 80px !important;
    padding: 0.75rem 1rem !important;
    line-height: 1.5 !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton button { border-radius: 10px; }

/* ── Metrics & captions ──────────────────────────────────────────────────── */
/* Captions carry real secondary copy (tab subtitles, event meta), so contrast
   matters. 0.88 keeps them visually secondary while staying readable on the
   dark background, where a lower opacity dips under WCAG AA. */
[data-testid="stMetricValue"] { font-size: 1.1rem; }
.stCaption, [data-testid="stCaptionContainer"] { opacity: 0.88; }

/* ── Event & set cards ───────────────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Genre chips ─────────────────────────────────────────────────────────── */
.ra-chip {
    display: inline-block; padding: 1px 9px; margin: 2px 4px 2px 0;
    border-radius: 999px; font-size: 0.73rem; line-height: 1.5;
    background: rgba(214,48,49,0.12); color: #FF9A9A;
    border: 1px solid rgba(214,48,49,0.28);
}
.ra-rank {
    display: inline-block; min-width: 1.5rem; text-align: center; padding: 0 6px;
    border-radius: 6px; font-weight: 700; background: #D63031; color: white; font-size: 0.8rem;
}

/* ── Expanders: tighter ──────────────────────────────────────────────────── */
.streamlit-expanderHeader { font-size: 0.83rem; }

/* ── Mobile ──────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container { padding-left: 0.75rem; padding-right: 0.75rem; max-width: 100%; }
    .stTabs [data-baseweb="tab"] { padding: 0.45rem 0.7rem; font-size: 0.8rem; }
    h1 { font-size: 1.6rem; }
    .stChatInput > div { border-radius: 10px !important; }
    /* On mobile, keep sidebar collapsed by default */
    section[data-testid="stSidebar"] { min-width: 0 !important; }
}

/* ── Auto-scroll chat to latest message ──────────────────────────────────── */
/* Scrolls the main content area to the bottom whenever a new message lands. */
.stChatMessage:last-child { scroll-margin-bottom: 80px; }
</style>
<script>
(function() {
    function scrollToBottom() {
        var main = window.parent.document.querySelector('section[data-testid="stMain"] .main');
        if (main) main.scrollTop = main.scrollHeight;
        var app = window.parent.document.querySelector('.appview-container');
        if (app) app.scrollTop = app.scrollHeight;
    }
    // Run on load and after short delay to catch Streamlit rerun rendering
    scrollToBottom();
    setTimeout(scrollToBottom, 200);
    setTimeout(scrollToBottom, 600);
})();
</script>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Knowledge-base warm-up (Streamlit Cloud cold start) ───────────────────────

@st.cache_resource(show_spinner=False)
def _ensure_kb_seeded() -> None:
    """Seed ChromaDB on first process start (no-op afterwards)."""
    from ingest import get_collection, ingest
    col = get_collection()
    if col.count() == 0:
        ingest()


# ── Session state ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    for tab in ("guide",):
        st.session_state.setdefault(f"messages_{tab}", [])
    st.session_state.setdefault("mix_sets", []) # list[{seed, setlist}]
    st.session_state.setdefault("intl_results", None) # last Beyond Berlin search
    st.session_state.setdefault("berlin_browse", None)  # {events, count}
    st.session_state.setdefault("total_cost", 0.0)
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("rated_events", set())


def _thread_key(tab: str) -> str:
    """Per-tab LangGraph thread so conversations don't cross-contaminate."""
    return f"{st.session_state['session_id']}:{tab}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Berlin Rave Atlas")
        st.caption("Your guide to Berlin's electronic music scene, and the rave map beyond it.")
        st.divider()

        st.markdown("**Model & voice**")
        model_names = [m["name"] for m in config.AVAILABLE_MODELS]
        model_ids = [m["id"] for m in config.AVAILABLE_MODELS]
        default_idx = model_ids.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in model_ids else 0

        if len(config.AVAILABLE_MODELS) == 1:
            # Single-model mode: no picker, just show the model name as a label.
            model_id = model_ids[0]
            st.caption(f"Model: **{model_names[0]}**", help="To enable more models, uncomment entries in config.py AVAILABLE_MODELS.")
        else:
            selected_name = st.selectbox("Model", model_names, index=default_idx,
                                         help="Haiku: fast and cheap (default). Gemini 2.5 Flash: Google model, strong reasoning and long context. GPT-4o Mini: OpenAI alternative, widely available. Mistral Large: non-Anthropic alternative. GPT-OSS 120B: open source, needs an Ollama key.")
            model_id = model_ids[model_names.index(selected_name)]

        tone = st.radio(
            "Tone",
            options=["Concise", "Detailed", "Expert"],
            index=0,
            horizontal=True,
            help=(
                "Concise = straight to the point. "
                "Detailed = full context and reasoning. "
                "Expert = insider level, labels, BPMs, scene history."
            ),
        )

        st.divider()
        with st.expander("How Berlin Rave Atlas works", expanded=False):
            st.markdown(
                "Your Berlin Guide is the main thing. Ask it anything and it will fetch "
                "real events, look up artists, dig into the knowledge base, or search the "
                "web depending on what you need. Berlin Raves is a quick browse when you "
                "already know what you want and just need to see the listings. Rave Set "
                "Builder builds a full playable set with previews and YouTube links. Raves "
                "Beyond Berlin is the same quick browse but for the rest of Europe. Set "
                "Tone above to control how the guide talks. Concise for short answers, "
                "Detailed for full context, Expert for label lineage and BPMs."
            )

        st.divider()
        st.caption("Session usage")
        col_tok, col_cost = st.columns(2)
        col_tok.metric("Tokens", f"{st.session_state['total_tokens']:,}")
        col_cost.metric("Cost", f"${st.session_state['total_cost']:.4f}")

        if st.button("Clear chat history", use_container_width=True):
            for tab in ("guide",):
                st.session_state[f"messages_{tab}"] = []
            st.session_state["mix_sets"] = []
            st.session_state["total_cost"] = 0.0
            st.session_state["total_tokens"] = 0
            st.session_state["rated_events"] = set()
            st.rerun()

        st.caption(f"Session `{st.session_state['session_id'][:8]}...`")

    return {"model_id": model_id, "tone": tone}


# ── Small shared renderers ────────────────────────────────────────────────────

def _render_tool_trace(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool trace, {len(tool_calls)} call(s)", expanded=False):
        for i, call in enumerate(tool_calls, 1):
            st.markdown(f"**{i}. `{call['name']}`**")
            if call.get("args"):
                st.json(call["args"], expanded=False)
            if call.get("output_preview"):
                st.code(call["output_preview"], language=None)
            if i < len(tool_calls):
                st.divider()


def _render_cost_badge(model: str, usage: dict, cost: float) -> None:
    st.caption(
        f"`{model}` · {usage.get('total_tokens', 0):,} tokens "
        f"({usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out) · ${cost:.5f}"
    )


def _chips(items: list[str]) -> str:
    return "".join(f'<span class="ra-chip">{i}</span>' for i in items if i)


def _sort_events(events: list[dict], sort_by: str, sort_dir: str) -> list[dict]:
    """
    Sort a flat event list by date or price.

    Price-unlisted events (price_numeric=None) always go to the end, regardless
    of direction, because their cost is unknown — not zero.
    """
    reverse = (sort_dir == "Desc")
    if sort_by == "Price":
        priced = [e for e in events if e.get("price_numeric") is not None]
        unlisted = [e for e in events if e.get("price_numeric") is None]
        return sorted(priced, key=lambda e: e["price_numeric"], reverse=reverse) + unlisted
    # Date sort
    return sorted(
        events,
        key=lambda e: (e.get("date", ""), e.get("start_time", "")),
        reverse=reverse,
    )


def _render_sort_controls(prefix: str) -> tuple[str, str]:
    """
    Single-dropdown sort above an event list.

    Returns (sort_by, sort_dir) where sort_by is "Date"|"Price"
    and sort_dir is "Asc"|"Desc".
    """
    _OPTIONS: list[tuple[str, str, str]] = [
        ("Newest first",       "Date",  "Desc"),
        ("Oldest first",       "Date",  "Asc"),
        ("Price, low to high", "Price", "Asc"),
        ("Price, high to low", "Price", "Desc"),
    ]
    labels = [o[0] for o in _OPTIONS]
    c1, _ = st.columns([3, 7])
    chosen = c1.selectbox(
        "Sort",
        labels,
        index=0,
        key=f"{prefix}_sort",
        label_visibility="collapsed",
    )
    _, sort_by, sort_dir = next(o for o in _OPTIONS if o[0] == chosen)
    return sort_by, sort_dir


def _event_meta_line(evt: dict) -> str:
    """One readable 'Sat 6 Jun, 22:30 to 06:00, Venue, Area' line for a card."""
    bits: list[str] = []
    # date_label / time_label are the cleaned fields from tools/events.py.
    # Fall back to the raw fields only if an older event dict lacks them.
    date_text = evt.get("date_label") or evt.get("date") or ""
    if date_text:
        bits.append(date_text)
    time_text = evt.get("time_label") or evt.get("start_time") or ""
    if time_text:
        bits.append(time_text)
    venue = evt.get("venue") or ""
    area = evt.get("area") or ""
    if venue:
        bits.append(venue + (f", {area}" if area and area != venue else ""))
    return " · ".join(bits)


def _render_event_card(
    evt: dict,
    *,
    rank: int | None = None,
    fit: str = "",
    rating_ctx: tuple[str, int] | None = None,
) -> None:
    """
    Render one event as a card with its Resident Advisor link always present.

    rating_ctx: (session_id, msg_idx) enables 👍/👎 feedback buttons (Weekend tab).
    """
    name = evt.get("name") or "Untitled event"
    url = evt.get("url") or ""
    with st.container(border=True):
        head_left, head_right = st.columns([8, 2])
        rank_badge = f'<span class="ra-rank">#{rank}</span> ' if rank else ""
        if url:
            head_left.markdown(f"{rank_badge}**[{name} ↗]({url})**", unsafe_allow_html=True)
        else:
            head_left.markdown(f"{rank_badge}**{name}**", unsafe_allow_html=True)

        # Price always carries its currency now (set in tools/events.py
        # _annotate_price): "Free", a real price like "€15", or "Price unlisted"
        # when RA's API returned no cost field (does not mean the event is free).
        price = evt.get("price") or "Price unlisted"
        head_right.markdown(f"**{price}**")

        meta = _event_meta_line(evt)
        if meta:
            st.caption(meta)
        address = evt.get("address") or ""
        if address:
            st.caption(address)
        if fit:
            st.markdown(fit)

        lineup = evt.get("lineup") or []
        genres = evt.get("genres") or []
        if genres:
            st.markdown(_chips(genres), unsafe_allow_html=True)
        if lineup:
            st.caption("Lineup: " + ", ".join(lineup[:8]) + ("..." if len(lineup) > 8 else ""))

        if rating_ctx is not None:
            _rating_buttons(evt, name, rank, *rating_ctx)


def _rating_buttons(evt: dict, name: str, rank: int | None, session_id: str, msg_idx: int) -> None:
    rated = st.session_state["rated_events"]
    rating_key = f"r_{msg_idx}_{name}"
    if rating_key in rated:
        st.caption("rated ✓")
        return
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("Like", key=f"up_{msg_idx}_{name}", help="Good fit, show me more like this"):
        memory.update_profile_from_feedback(session_id, evt, liked=True)
        rated.add(rating_key)
        st.toast("Liked, profile updated")
        st.rerun()
    if c2.button("Not for me", key=f"dn_{msg_idx}_{name}", help="Not my taste"):
        memory.update_profile_from_feedback(session_id, evt, liked=False)
        rated.add(rating_key)
        st.toast("Noted, will avoid similar events")
        st.rerun()


# ── Events block (Weekend tab): unified cards + links + ranking + ratings ─────

def _parse_tool_output(tool_calls: list[dict], name: str) -> Any:
    call = next((c for c in tool_calls if c["name"] == name), None)
    if not call or not call.get("full_output"):
        return None
    try:
        return json.loads(call["full_output"])
    except (json.JSONDecodeError, TypeError):
        return None


def _render_events_block(tool_calls: list[dict], msg_idx: int) -> None:
    """
    Render real event cards from a find_events call (every card carries its RA
    link). If compare_events also ran, sort by its ranking, show fit summaries,
    and offer 👍/👎 feedback.
    """
    raw_events = _parse_tool_output(tool_calls, "find_events")
    if raw_events is None:
        return # find_events wasn't called on this turn
    if not isinstance(raw_events, list) or not raw_events:
        st.info("No Resident Advisor events came back for that window. Try a different date or filter.")
        return

    ranked = _parse_tool_output(tool_calls, "compare_events") or {}
    ranked_list = ranked.get("ranked_events", []) if isinstance(ranked, dict) else []
    rank_by_name = {r.get("event_name"): r for r in ranked_list}

    # Order by rank when available; unranked events keep their original order after.
    def sort_key(item):
        r = rank_by_name.get(item.get("name"))
        return r.get("rank", 9_999) if r else 9_999
    ordered = sorted(raw_events, key=sort_key) if ranked_list else raw_events

    session_id = st.session_state["session_id"]
    st.markdown("**Events from Resident Advisor**")
    for evt in ordered[:8]:
        r = rank_by_name.get(evt.get("name"))
        _render_event_card(
            evt,
            rank=r.get("rank") if r else None,
            fit=(r.get("fit_summary", "") if r else ""),
            rating_ctx=(session_id, msg_idx),
        )


# ── Set-list card (Rave Set Builder) ──────────────────────────────────────────

def _energy_bar(energy: int) -> str:
    """Render an energy value as a mini visual bar (e.g. ████░░ 6/10)."""
    filled = round(energy / 2) # out of 5 blocks for compactness
    empty = 5 - filled
    return "█" * filled + "░" * empty + f" {energy}/10"


_ROLE_EMOJI: dict[str, str] = {
    "opener":     "🌅",
    "build":      "📈",
    "peak":       "🔥",
    "sustain":    "⚡",
    "resolution": "🌊",
    "closer":     "🌙",
}


def _role_badge(role: str) -> str:
    """Return a short emoji+label string for a track's arc role."""
    role_lower = role.lower().strip()
    emoji = _ROLE_EMOJI.get(role_lower, "")
    return f"{emoji} {role.capitalize()}" if role_lower else ""


def _render_setlist(sl: dict) -> None:
    tracks = sl.get("tracks", [])
    if not tracks:
        st.warning("Couldn't build a playable set just now. The music API or AI may be busy, try again.")
        return

    st.markdown(f"### {sl.get('title', 'Set')}")
    arc = sl.get("energy_arc", [])
    if arc:
        st.caption(
            f"Arc: **{' → '.join(str(e) for e in arc)}** "
            f"({len(tracks)} tracks, ~{len(tracks) * 4} min)"
        )

    # Set story — the 2-3 sentence arc narrative from Pass 2
    story = sl.get("set_story", "")
    if story:
        st.info(story, icon="🎧")

    # ── Summary table ─────────────────────────────────────────────────────────
    with st.expander("Track list at a glance", expanded=True):
        rows = []
        for i, t in enumerate(tracks, 1):
            energy = t.get("energy", 5)
            show_artist = t.get("llm_artist") or t.get("artist", "")
            show_title = t.get("llm_title") or t.get("title", "")
            role = _role_badge(t.get("role", ""))
            bpm_val = t.get("bpm") or t.get("bpm_target")
            bpm_str = f"~{bpm_val}" if bpm_val else "-"
            rows.append(
                f"| **{i}** | {show_artist} | *{show_title}* | {role} | {_energy_bar(energy)} | {bpm_str} |"
            )

        header = "| # | Artist | Track | Role | Energy | BPM |"
        sep    = "|---|--------|-------|------|--------|-----|"
        st.markdown(header + "\n" + sep + "\n" + "\n".join(rows), unsafe_allow_html=False)

    # ── Playable track cards ──────────────────────────────────────────────────
    for i, t in enumerate(tracks, 1):
        is_fallback = t.get("deezer_fallback", False)
        llm_artist  = t.get("llm_artist") or ""
        llm_title   = t.get("llm_title") or ""

        with st.container(border=True):
            c_num, c_info, c_nrg, c_meta = st.columns([1, 8, 2, 2])
            c_num.markdown(f"**{i}**")

            if is_fallback and llm_title:
                # Show the original LLM pick as the headline so the reason still
                # makes sense. The Deezer substitute appears in the preview label.
                c_info.markdown(f"**{llm_artist}** · *{llm_title}*")
            else:
                c_info.markdown(f"**{t.get('artist', '')}** · *{t.get('title', '')}*")

            c_nrg.markdown(_energy_bar(t.get("energy", 5)))

            # Role badge + BPM
            role_str = _role_badge(t.get("role", ""))
            bpm_display = t.get("bpm") or t.get("bpm_target")
            meta_parts = []
            if role_str:
                meta_parts.append(role_str)
            if bpm_display:
                meta_parts.append(f"~{bpm_display} BPM")
            if meta_parts:
                c_meta.caption("  \n".join(meta_parts))

            if t.get("reason"):
                st.caption(t["reason"])

            links: list[str] = []
            if t.get("deezer_url"):
                links.append(f"[Deezer ↗]({t['deezer_url']})")
            if t.get("youtube_url"):
                yt_label = "YouTube ↗" if t.get("youtube_verified") else "YouTube (search) ↗"
                links.append(f"[{yt_label}]({t['youtube_url']})")
            if links:
                st.caption(" · ".join(links))

            if t.get("preview_url"):
                if is_fallback and llm_title:
                    st.caption(
                        f"Preview: **{t.get('artist', '')}** - *{t.get('title', '')}* "
                        f"(closest available on Deezer — use YouTube for the original)"
                    )
                else:
                    st.caption("30-second preview")
                st.audio(t["preview_url"], format="audio/mp3")
            elif is_fallback and not t.get("preview_url"):
                st.caption("No Deezer preview available — use the YouTube link above.")



# ── Chat message renderer (Weekend / Learn) ───────────────────────────────────

def _render_chat(tab_key: str, empty_hint: str, show_event_cards: bool = True) -> None:
    messages = st.session_state[f"messages_{tab_key}"]
    if not messages:
        st.caption(empty_hint)
        return
    # Event cards are rendered once, for the first assistant message that
    # called find_events (only when show_event_cards=True). Follow-up messages
    # never re-render the same cards; the agent's text has RA links inline.
    events_rendered_at: int | None = None
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.warning(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and not msg.get("blocked"):
                tc = msg.get("tool_calls", [])
                if tab_key == "guide" and show_event_cards:
                    has_events = any(c.get("name") == "find_events" for c in tc)
                    if has_events and events_rendered_at is None:
                        _render_events_block(tc, idx)
                        events_rendered_at = idx
                _render_tool_trace(tc)
                if msg.get("usage"):
                    _render_cost_badge(msg.get("model", ""), msg["usage"], msg.get("cost_estimate", 0.0))


def _handle_chat_input(tab_key: str, settings: dict, placeholder: str) -> None:
    prompt = st.chat_input(placeholder, key=f"chat_input_{tab_key}")
    # Pick up an auto-submit queued by the "Discuss these events" button
    auto_key = f"auto_submit_{tab_key}"
    if not prompt and auto_key in st.session_state:
        prompt = st.session_state.pop(auto_key)
    if not prompt:
        return
    messages = st.session_state[f"messages_{tab_key}"]
    messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        result = run_agent(
            message=prompt,
            session_id=st.session_state["session_id"],
            model_id=settings["model_id"],
            tone=settings["tone"],
            thread_key=_thread_key(tab_key),
        )
    messages.append({
        "role": "assistant",
        "content": result["text"],
        "blocked": result["blocked"],
        "tool_calls": result["tool_calls"],
        "usage": result["usage"],
        "cost_estimate": result["cost_estimate"],
        "model": result["model"],
    })
    st.session_state["total_cost"] += result["cost_estimate"]
    st.session_state["total_tokens"] += result["usage"]["total_tokens"]
    st.rerun()


# ── Digest section (Berlin only) ──────────────────────────────────────────────

_DIGEST_STALE_HOURS = 20  # auto-refresh if the stored digest is older than this


def _render_digest_section() -> None:
    session_id = st.session_state["session_id"]

    # Load digest + age for whichever scope has a stored one.
    digest, age_hours = memory.load_digest_with_age(session_id)
    if digest is None:
        digest, age_hours = memory.load_digest_with_age("__global__")

    # Auto-refresh once per browser session when the stored digest is stale.
    # This covers the reviewer-on-Monday scenario: the Friday digest is >60h
    # old, so we silently regenerate rather than showing last week's picks.
    if (
        digest is not None
        and age_hours is not None
        and age_hours > _DIGEST_STALE_HOURS
        and not st.session_state.get("_digest_auto_refreshed")
    ):
        st.session_state["_digest_auto_refreshed"] = True
        with st.spinner("Refreshing picks for today..."):
            fresh = generate_digest(session_id)
        if fresh:
            digest = fresh
            age_hours = 0.0

    label = "What's worth going to this week" + ("" if digest else " (generate now)")
    with st.expander(label, expanded=bool(digest)):
        if digest:
            st.markdown(digest)
            # Show freshness stamp so the user knows how recent the picks are.
            if age_hours is not None:
                if age_hours < 1:
                    age_str = "just now"
                elif age_hours < 2:
                    age_str = "1 hour ago"
                else:
                    age_str = f"{int(age_hours)} hours ago"
                st.caption(f"Updated {age_str} · Tap **Get fresh picks** to regenerate")
        else:
            st.caption(
                "The agent hasn't picked yet. Hit the button to get AI-curated "
                "top events for this weekend, personalised to your taste profile."
            )
        if st.button("Get fresh picks", key="btn_regen_digest"):
            st.session_state["_digest_auto_refreshed"] = True  # don't double-fire
            with st.spinner("Fetching Berlin events and writing your picks..."):
                new_digest = generate_digest(session_id)
            if new_digest:
                st.rerun()
            else:
                st.warning("Could not generate picks. Resident Advisor or the AI may be busy.")


# ── Berlin raw browse ─────────────────────────────────────────────────────────

def _render_berlin_browse() -> None:
    col_count, col_from, col_to, col_browse, col_clear = st.columns([1, 1.5, 1.5, 1, 1])
    count = col_count.selectbox("Show", [5, 10, 25, 50], index=1, key="browse_count", label_visibility="collapsed")
    today = date.today()
    browse_from = col_from.date_input("From", value=today, key="browse_from", label_visibility="collapsed")
    browse_to = col_to.date_input("To", value=today + timedelta(days=3), key="browse_to", label_visibility="collapsed")
    if col_browse.button("Browse", type="primary", key="btn_browse_berlin"):
        from tools.events import find_events as _find
        with st.spinner("Fetching Berlin events..."):
            events = _find(browse_from.isoformat(), browse_to.isoformat())
        st.session_state["berlin_browse"] = {"events": events[:count], "count": count}
        st.rerun()
    if st.session_state.get("berlin_browse"):
        if col_clear.button("Clear", key="btn_clear_berlin"):
            st.session_state["berlin_browse"] = None
            st.rerun()

    res = st.session_state.get("berlin_browse")
    if res:
        events = res["events"]
        if not events:
            st.info("No Resident Advisor listings in that window. Try widening the dates.")
        else:
            st.caption(f"{len(events)} parties on RA.")
            sort_by, sort_dir = _render_sort_controls("berlin_browse")
            for evt in _sort_events(events, sort_by, sort_dir):
                _render_event_card(evt)


# ── Tabs ──────────────────────────────────────────────────────────────────────


def _tab_guide(settings: dict) -> None:
    st.info(
        "Ask me anything. What's on this Friday, which night matches your taste, "
        "what Berghain's door is actually like, whether you need cash, what minimal "
        "techno sounds like, the difference between Ostgut Ton and Klockworks, how to "
        "stay safe on a long night, the story behind Tresor, what to wear, when to arrive. "
        "I know this scene. Parties, music history, genre theory, rave culture, harm reduction, "
        "all of it. Just ask.",
        icon="🎛️",
    )
    st.divider()
    _render_chat("guide", "Ask anything about Berlin nights, music, or scene culture.")
    _handle_chat_input(
        "guide",
        settings,
        placeholder="What's on this Friday? / Tell me about Berghain / What is minimal techno?",
    )


def _tab_parties(settings: dict) -> None:  # noqa: ARG001
    st.info(
        "Live Resident Advisor listings for Berlin. Pick your dates, filter by price or genre, "
        "and see what's on. No chat needed.",
        icon="🗓️",
    )
    st.divider()
    _render_digest_section()
    st.divider()
    _render_berlin_browse()


def _tab_mix_builder(settings: dict) -> None:
    st.info("Describe a slot or mood and get a full 1-hour set with a warm-up to peak arc, 30-second previews, and YouTube links.", icon="🎛️")
    st.caption(
        "Be specific for best results. 'Hypnotic 130 BPM 2am Berghain' beats 'techno'. "
        "Try a walk-in set, a peak-time set, or a Sunday comedown."
    )
    st.divider()
    sets = st.session_state["mix_sets"]
    if not sets:
        st.info("Describe a slot or a mood below and Berlin Rave Atlas builds you a playable 1-hour set.")
    for entry in sets:
        with st.chat_message("user"):
            st.markdown(entry["seed"])
        with st.chat_message("assistant"):
            _render_setlist(entry["setlist"])

    if sets:
        st.caption("Want a different set? Just describe another slot below to build a fresh one.")

    seed = st.chat_input(
        "Warm-up for KitKat Saturday 23h / Hypnotic 2am Berghain set / Sunday Sisyphos comedown",
        key="chat_input_mix",
    )
    if seed:
        with st.spinner("Building your set and fetching previews, about 30 seconds for 16 tracks..."):
            sl = build_setlist(seed=seed, n=16, model_id=settings["model_id"])
        st.session_state["mix_sets"].append({"seed": seed, "setlist": sl})
        st.rerun()


def _parallel_city_search(
    cities: list[str],
    date_from: str,
    date_to: str,
    filters: dict | None,
) -> dict[str, list]:
    """
    Fetch RA events for every city in the list concurrently.
    Returns {city: [events]}, preserving all cities (empty list when none found).
    Cities that error return an empty list silently.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tools.events import find_events as _find

    results: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_find, date_from, date_to, filters=filters, city=c): c
            for c in cities
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                results[c] = future.result() or []
            except Exception:
                results[c] = []
    return results


# Top RA-active European cities used when "All Europe" is selected without a
# specific city. Capped at 30 to keep parallel search time under ~15 seconds.
_RA_HOTSPOTS: list[str] = [
    "Amsterdam", "London", "Paris", "Barcelona", "Belgrade",
    "Vienna", "Zurich", "Copenhagen", "Lisbon", "Prague",
    "Budapest", "Warsaw", "Brussels", "Rotterdam", "Hamburg",
    "Frankfurt", "Cologne", "Manchester", "Dublin", "Milan",
    "Tbilisi", "Stockholm", "Oslo", "Helsinki", "Athens",
    "Istanbul", "Edinburgh", "Glasgow", "Bologna", "Madrid",
]


def _tab_beyond_berlin(settings: dict) -> None:
    st.info(
        "Live Resident Advisor listings for any European city or region. "
        "Pick a region and optionally narrow to one city.",
        icon="🌍",
    )
    st.caption(
        "Best coverage: Amsterdam, Paris, Barcelona, Belgrade, Vienna, Zurich, Copenhagen. "
        "Leave City empty to search the whole region. For Berlin, use Your Berlin Guide."
    )
    st.divider()

    # ── Region + city selectors ───────────────────────────────────────────────
    col_region, col_city = st.columns([2, 3])

    region_names = list(config.CITY_REGIONS.keys())
    region = col_region.selectbox(
        "Region",
        region_names,
        index=0,
        help="Pick a region. Leave City empty to search all cities in the region.",
    )
    region_cities = config.CITY_REGIONS.get(region) or []
    city_list = region_cities if region_cities else config.AVAILABLE_CITIES
    city_list = [c for c in city_list if c != "Berlin"]
    city = col_city.selectbox(
        "City (optional)",
        city_list,
        index=None,
        placeholder="All cities in region...",
        help="Leave empty to search the whole region at once.",
    )

    # ── Date range ────────────────────────────────────────────────────────────
    col_from, col_to = st.columns(2)
    today = date.today()
    date_from = col_from.date_input("From", value=today)
    date_to = col_to.date_input("To", value=today + timedelta(days=3))

    # ── Optional filters ──────────────────────────────────────────────────────
    col_genre, col_price = st.columns([3, 2])
    genres = col_genre.multiselect(
        "Genres (optional)",
        [
            "Techno", "House", "Minimal", "Tech House", "Deep House",
            "Progressive House", "Melodic House & Techno", "Trance", "Psytrance",
            "Drum & Bass", "Jungle", "UK Garage", "Breaks", "Dubstep",
            "Ambient", "Experimental", "Electronica", "Noise",
            "Disco", "Electro", "Afro House", "Bass Music",
            "Hardcore", "Industrial", "EBM", "Dark Techno",
        ],
        help="Filters by RA genre tags. When searching a whole region, add a genre to narrow results.",
    )
    max_price = col_price.slider("Max price (€, 0 = no limit)", 0, 80, 0, 5)

    col_search, col_clear = st.columns([3, 1])
    if col_search.button("Find rave parties on RA", type="primary", use_container_width=True):
        if not city and region == "All Europe" and not genres:
            st.error(
                "Too broad: pick a city, select a specific region, or add a genre filter "
                "before searching all of Europe."
            )
        elif date_to < date_from:
            st.error("'To' date must be on or after 'From' date.")
        else:
            filters: dict[str, Any] = {}
            if genres:
                filters["genres"] = genres
            if max_price > 0:
                filters["max_price"] = float(max_price)
            from tools.events import find_events as _find

            if city:
                # Single-city search — existing behaviour
                with st.spinner(f"Querying Resident Advisor for {city}..."):
                    events = _find(
                        date_from.isoformat(),
                        date_to.isoformat(),
                        filters=filters or None,
                        city=city,
                    )
                st.session_state["intl_results"] = {
                    "label": city,
                    "city": city,
                    "events": events,
                    "by_city": None,
                }
            else:
                # Region-wide parallel search
                if region == "All Europe":
                    cities_to_search = _RA_HOTSPOTS
                    label = f"All Europe ({', '.join(genres)})" if genres else "All Europe"
                else:
                    cities_to_search = city_list  # already excludes Berlin
                    genre_note = f" ({', '.join(genres)})" if genres else ""
                    label = f"{region}{genre_note}"
                n = len(cities_to_search)
                genre_desc = ", ".join(genres) if genres else "all genres"
                with st.spinner(f"Searching {n} cities in {region} for {genre_desc}..."):
                    by_city = _parallel_city_search(
                        cities_to_search,
                        date_from.isoformat(),
                        date_to.isoformat(),
                        filters or None,
                    )
                all_events = sorted(
                    [e for evts in by_city.values() for e in evts],
                    key=lambda e: (e.get("date", ""), e.get("start_time", "")),
                )
                st.session_state["intl_results"] = {
                    "label": label,
                    "city": None,
                    "events": all_events,
                    "by_city": by_city,
                }
            st.rerun()

    if st.session_state.get("intl_results"):
        if col_clear.button("Clear results", use_container_width=True):
            st.session_state["intl_results"] = None
            st.rerun()

    res = st.session_state.get("intl_results")
    if res:
        events = res["events"]
        st.divider()
        if not events:
            st.info(
                f"No Resident Advisor listings found for **{res['label']}** in that date range. "
                "Try widening the dates or adding a different genre. "
                "Amsterdam, Paris, Barcelona, Belgrade, and Zurich tend to have the most listings."
            )
        elif res.get("by_city"):
            # Region mode: group by city, sorted by number of events descending
            cities_with_results = {
                c: evts for c, evts in res["by_city"].items() if evts
            }
            n_cities = len(cities_with_results)
            st.markdown(
                f"**{len(events)} parties across {n_cities} {'city' if n_cities == 1 else 'cities'}** "
                f"in {res['label']} via Resident Advisor"
            )
            sort_by, sort_dir = _render_sort_controls("beyond_berlin")
            expand_all = n_cities <= 3
            for city_name, city_evts in sorted(
                cities_with_results.items(), key=lambda x: -len(x[1])
            ):
                with st.expander(f"{city_name} ({len(city_evts)})", expanded=expand_all):
                    for evt in _sort_events(city_evts, sort_by, sort_dir)[:15]:
                        _render_event_card(evt)
        else:
            # Single-city mode
            city_label = res.get("city") or res.get("label", "")
            st.markdown(f"**{len(events)} rave parties in {city_label}** via Resident Advisor")
            sort_by, sort_dir = _render_sort_controls("beyond_berlin")
            for evt in _sort_events(events, sort_by, sort_dir)[:30]:
                _render_event_card(evt)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Berlin Rave Atlas", page_icon="🎛️", layout="wide",
                       initial_sidebar_state="expanded")
    _inject_css()
    # Show a centred loading screen on the first cold start while ChromaDB seeds.
    # cache_resource means this only ever runs once per process; subsequent page
    # loads skip it entirely and the placeholder is never shown.
    from ingest import get_collection
    _needs_seed = get_collection().count() == 0
    if _needs_seed:
        with st.spinner(""):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(
                    "<div style='text-align:center; padding: 4rem 0;'>"
                    "<div style='font-size:2.5rem; margin-bottom:1rem;'>🎛️</div>"
                    "<h2 style='margin-bottom:0.5rem;'>Building knowledge base</h2>"
                    "<p style='color:gray;'>First run only — this takes about 30 seconds.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            _ensure_kb_seeded()
    else:
        _ensure_kb_seeded()
    _init_session()
    get_scheduler() # local convenience; production automation runs via GitHub Actions

    settings = _render_sidebar()
    # Pin every llm_client.chat() call this rerun to the sidebar-selected model.
    # This covers compare_events, explain_music internals, and any other tool
    # that calls llm_client.chat() without an explicit model argument.
    # build_setlist passes model_id explicitly and that value takes precedence.
    llm_client.set_session_model(settings["model_id"])

    st.title("Berlin Rave Atlas")
    st.caption(
        "Your guide to Berlin's electronic music scene. Find the right night, learn "
        "the music, build a set, and browse raves across Europe."
    )

    tab_guide, tab_parties, tab_mix, tab_beyond = st.tabs(
        ["Your Berlin Guide", "Berlin Raves", "Rave Set Builder", "Raves Beyond Berlin"]
    )
    with tab_guide:
        _tab_guide(settings)
    with tab_parties:
        _tab_parties(settings)
    with tab_mix:
        _tab_mix_builder(settings)
    with tab_beyond:
        _tab_beyond_berlin(settings)


if __name__ == "__main__":
    main()
