"""
Berlin Rave Atlas, Streamlit UI.

Four tabs, a Berlin-first agent, and one deliberate non-agent browser.

  Raves in Berlin        Berlin event discovery, compare, ratings (ReAct agent)
  Rave Set Builder       1-hour set builder (direct, deterministic tool call)
  Rave Wiki              KB chat grounded in the knowledge base (ReAct agent, RAG)
  Beyond Berlin          browse all European Resident Advisor cities (plain API browse)

Why the split. The agent earns its cost where runtime reasoning matters
(planning a Berlin night across several tools). Rave Set Builder is a single
focused artefact, so it calls build_setlist directly, which guarantees every
set is fully enriched with Deezer previews and YouTube links every time. Beyond
Berlin is pure retrieval, so it is an honest, fast browse rather than a chat.

Chat. Each tab keeps its own message list and its own agent conversation
thread, so a Wiki question never bleeds into the Berlin planner's context.
Messages render oldest to newest directly in the page; Streamlit's chat_input is
sticky at the bottom. On submit we append and rerun, so everything renders
through one path.

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
    for tab in ("weekend", "learn"):
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
            options=["concise", "elaborated", "expert"],
            index=0,
            horizontal=True,
            help=(
                "concise = straight to the point. "
                "elaborated = full context and reasoning. "
                "expert = insider level, labels, BPMs, scene history."
            ),
        )

        with st.expander("Advanced, response sampling", expanded=False):
            st.caption(
                "Two dials that change how the AI writes. Leave them as they are "
                "unless you want to experiment."
            )
            st.caption(
                "**Temperature** controls creativity. Low (near 0) makes answers "
                "focused and repeatable; high makes them more varied and surprising. "
                "0.7 is a balanced default."
            )
            st.caption(
                "**Top-p** controls how wide a vocabulary the AI draws from. At 1.0 "
                "it considers all word options; lower values keep it to the most "
                "likely words, which makes writing safer and more predictable. "
                "Leave it at 1.0 unless answers feel too loose."
            )
            temperature = st.slider("Temperature (creativity)", 0.0, 1.5, 0.7, 0.05)
            top_p = st.slider("Top-p (vocabulary breadth)", 0.5, 1.0, 1.0, 0.05)

        st.divider()
        with st.expander("How Berlin Rave Atlas works", expanded=False):
            st.markdown(
                "**Four tabs, one agent, a curated knowledge base.**\n\n"
                "- **Plan Your Night** fetches live RA events, ranks them by your taste, "
                "and links every card to its RA page. Rate picks to teach it what you like.\n"
                "- **Rave Set Builder** takes a brief (slot, vibe, BPM) and builds a "
                "playable 1-hour set with 30-second previews and YouTube links.\n"
                "- **Learn the Scene** answers anything about genres, labels, scene "
                "history, rave etiquette, and harm reduction from the curated knowledge "
                "base. It falls back to web search when needed.\n"
                "- **Beyond Berlin** browses live RA listings for any European city.\n\n"
                "Set **Tone** above to control depth. concise for quick answers, "
                "elaborated for full context, expert for label lineage and BPMs."
            )

        st.divider()
        st.caption("Session usage")
        col_tok, col_cost = st.columns(2)
        col_tok.metric("Tokens", f"{st.session_state['total_tokens']:,}")
        col_cost.metric("Cost", f"${st.session_state['total_cost']:.4f}")

        if st.button("Clear chat history", use_container_width=True):
            for tab in ("weekend", "learn"):
                st.session_state[f"messages_{tab}"] = []
            st.session_state["mix_sets"] = []
            st.session_state["total_cost"] = 0.0
            st.session_state["total_tokens"] = 0
            st.session_state["rated_events"] = set()
            st.rerun()

        st.caption(f"Session `{st.session_state['session_id'][:8]}...`")

    return {"model_id": model_id, "tone": tone, "temperature": temperature, "top_p": top_p}


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
        # _annotate_price), and is never blank: it is "Free" or "No price listed".
        price = evt.get("price") or "No price listed"
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


def _render_setlist(sl: dict) -> None:
    tracks = sl.get("tracks", [])
    if not tracks:
        st.warning("Couldn't build a playable set just now. The music API or AI may be busy, try again.")
        return

    st.markdown(f"### {sl.get('title', 'Set')}")
    arc = sl.get("energy_arc", [])
    if arc:
        st.caption(f"Arc: **{' → '.join(str(e) for e in arc)}** ({len(tracks)} tracks, ~{len(tracks) * 4} min)")

    # ── Summary table (quick overview at a glance) ────────────────────────────
    with st.expander("Track list at a glance", expanded=True):
        rows = []
        for i, t in enumerate(tracks, 1):
            energy = t.get("energy", 5)
            rows.append(
                f"| **{i}** | {t.get('artist', '')} | *{t.get('title', '')}* | "
                f"{_energy_bar(energy)} |"
            )
        st.markdown(
            "| # | Artist | Track | Energy |\n"
            "|---|--------|-------|--------|\n"
            + "\n".join(rows),
            unsafe_allow_html=False,
        )

    # ── Playable track cards ──────────────────────────────────────────────────
    for i, t in enumerate(tracks, 1):
        with st.container(border=True):
            c_num, c_info, c_nrg = st.columns([1, 9, 2])
            c_num.markdown(f"**{i}**")
            c_info.markdown(f"**{t.get('artist', '')}** · *{t.get('title', '')}*")
            c_nrg.markdown(_energy_bar(t.get("energy", 5)))

            if t.get("reason"):
                st.caption(t["reason"])

            links = []
            if t.get("deezer_url"):
                links.append(f"[Deezer ↗]({t['deezer_url']})")
            if t.get("youtube_url"):
                yt_label = "YouTube ↗" if t.get("youtube_verified") else "YouTube (search) ↗"
                links.append(f"[{yt_label}]({t['youtube_url']})")
            if links:
                st.caption(" · ".join(links))

            if t.get("preview_url"):
                if t.get("deezer_fallback"):
                    st.caption(
                        f"30-second preview of another track by "
                        f"**{t.get('artist', '')}** (exact track not on Deezer, "
                        f"use the YouTube link for the real one)"
                    )
                else:
                    st.caption("30-second preview")
                st.audio(t["preview_url"], format="audio/mp3")


# ── Chat message renderer (Weekend / Learn) ───────────────────────────────────

def _render_chat(tab_key: str, empty_hint: str) -> None:
    messages = st.session_state[f"messages_{tab_key}"]
    if not messages:
        st.caption(empty_hint)
        return
    # Event cards are rendered once, for the first assistant message that
    # called find_events. Follow-up messages never re-render the same cards;
    # the agent's text already references events by name with RA links.
    events_rendered_at: int | None = None
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.warning(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and not msg.get("blocked"):
                tc = msg.get("tool_calls", [])
                if tab_key == "weekend":
                    has_events = any(c.get("name") == "find_events" for c in tc)
                    if has_events and events_rendered_at is None:
                        _render_events_block(tc, idx)
                        events_rendered_at = idx
                    # Subsequent turns: event cards already visible above, skip.
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
            temperature=settings["temperature"],
            top_p=settings["top_p"],
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

def _render_digest_section() -> None:
    session_id = st.session_state["session_id"]
    digest = memory.load_digest(session_id) or memory.load_digest("__global__")
    label = "Agent's Top Picks" + ("" if digest else " — tap to generate")
    with st.expander(label, expanded=bool(digest)):
        if digest:
            st.markdown(digest)
        else:
            st.caption("The agent hasn't picked yet. Hit the button to get AI-curated top events for this weekend, personalised to your taste profile.")
        if st.button("Get fresh picks", key="btn_regen_digest"):
            with st.spinner("Fetching Berlin events and writing your picks..."):
                new_digest = generate_digest(session_id)
            if new_digest:
                st.rerun()
            else:
                st.warning("Could not generate picks. Resident Advisor or the AI may be busy.")


# ── Berlin raw browse ─────────────────────────────────────────────────────────

def _render_berlin_browse() -> None:
    st.markdown("#### Browse Berlin parties")
    col_count, col_from, col_to, col_btn = st.columns([1, 1.5, 1.5, 1])
    count = col_count.selectbox("Show", [5, 10, 25, 50], index=1, key="browse_count", label_visibility="collapsed")
    today = date.today()
    browse_from = col_from.date_input("From", value=today, key="browse_from", label_visibility="collapsed")
    browse_to = col_to.date_input("To", value=today + timedelta(days=3), key="browse_to", label_visibility="collapsed")
    if col_btn.button("Browse", type="primary", key="btn_browse_berlin"):
        from tools.events import find_events as _find
        with st.spinner("Fetching Berlin events..."):
            events = _find(browse_from.isoformat(), browse_to.isoformat())
        st.session_state["berlin_browse"] = {"events": events[:count], "count": count}
        st.rerun()

    res = st.session_state.get("berlin_browse")
    if res:
        events = res["events"]
        if not events:
            st.info("No Resident Advisor listings in that window. Try widening the dates.")
        else:
            st.caption(f"{len(events)} parties on RA — switch to Chat to discuss any of them with the agent.")
            for evt in events:
                _render_event_card(evt)


# ── Tabs ──────────────────────────────────────────────────────────────────────


def _tab_weekend(settings: dict) -> None:
    sub_find, sub_chat = st.tabs(["Find Parties", "Chat with Agent"])

    with sub_find:
        st.caption("Browse raw RA listings for any date window, then let the agent pick the best ones for you.")
        _render_berlin_browse()
        st.divider()
        _render_digest_section()

    with sub_chat:
        browse_res = st.session_state.get("berlin_browse")
        if browse_res and browse_res.get("events"):
            events = browse_res["events"]
            preview_names = [e.get("name", "") for e in events[:3] if e.get("name")]
            preview = ", ".join(preview_names) + ("..." if len(events) > 3 else "")
            col_info, col_btn = st.columns([3, 1])
            col_info.info(f"Browsing **{len(events)} parties** — {preview}", icon="🗂️")
            if col_btn.button("Discuss these events", key="btn_discuss_events"):
                event_lines = "\n".join(
                    f"- {e.get('name','')} at {e.get('venue','')} ({e.get('date_label','')}, {e.get('price','')})"
                    for e in events[:10]
                )
                msg = (
                    f"I've been browsing {len(events)} Berlin parties on RA. Here's the list:\n"
                    f"{event_lines}\n\n"
                    "Can you help me pick the best one and tell me more about the headliners or the sound?"
                )
                st.session_state["auto_submit_weekend"] = msg
                st.rerun()
        _render_chat("weekend", "What's on in Berlin this weekend? Ask about any date, genre, or budget.")
        _handle_chat_input(
            "weekend", settings,
            placeholder="What's on this Friday? / Find me a night at Tresor or Sisyphos / Who's playing at Berghain?",
        )


def _tab_learn(settings: dict) -> None:
    st.info("Ask about genres, labels, scene history, rave culture, or harm reduction. This tab is purely for learning — for finding real events, use the Plan Your Night tab.", icon="📖")
    st.caption(
        "Answers are grounded in the curated knowledge base. When something falls outside "
        "it, the agent searches the web and tells you so."
    )
    st.divider()
    _render_chat("learn", "Ask anything about electronic music, Berlin's scene, or rave culture.")
    _handle_chat_input("learn", settings,
                       placeholder="What is minimal techno? / History of Tresor / What should I bring to a rave?")


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
            sl = build_setlist(seed=seed, n=16)
        st.session_state["mix_sets"].append({"seed": seed, "setlist": sl})
        st.rerun()


def _tab_beyond_berlin(settings: dict) -> None:
    st.info("Browse live Resident Advisor listings for any European city. Pick a region, a city, and a date range. No chat, just events.", icon="🌍")
    st.caption(
        "Best coverage: Amsterdam, Paris, Barcelona, Belgrade, Vienna, Zurich, Copenhagen. "
        "Smaller cities may return few results. For Berlin recommendations and agent chat, use Raves in Berlin."
    )
    st.divider()

    # ── Region + city selectors ───────────────────────────────────────────────
    col_region, col_city = st.columns([2, 3])

    region_names = list(config.CITY_REGIONS.keys())
    region = col_region.selectbox(
        "Region",
        region_names,
        index=0,
        help="Pick a region to filter the city list.",
    )
    region_cities = config.CITY_REGIONS.get(region) or []
    city_list = region_cities if region_cities else config.AVAILABLE_CITIES
    city_list = [c for c in city_list if c != "Berlin"]
    city = col_city.selectbox("City", city_list, index=None, placeholder="Select a city...")

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
        help="Filters returned listings by RA's genre tags. Leave empty for all genres.",
    )
    max_price = col_price.slider("Max price (€, 0 = no limit)", 0, 80, 0, 5)

    col_search, col_clear = st.columns([3, 1])
    if col_search.button("Find rave parties on RA", type="primary", use_container_width=True):
        if not city:
            st.error("Pick a city first.")
        elif date_to < date_from:
            st.error("'To' date must be on or after 'From' date.")
        else:
            filters: dict[str, Any] = {}
            if genres:
                filters["genres"] = genres
            if max_price > 0:
                filters["max_price"] = float(max_price)
            from tools.events import find_events as _find
            with st.spinner(f"Querying Resident Advisor for {city}..."):
                events = _find(
                    date_from.isoformat(),
                    date_to.isoformat(),
                    filters=filters or None,
                    city=city,
                )
            st.session_state["intl_results"] = {"city": city, "events": events}
            st.rerun()
    # Clear button only shown when results are present; without it there is no
    # way to dismiss the results and start a fresh search from a clean state.
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
                f"No Resident Advisor listings found for **{res['city']}** in that date range. "
                "Try widening the dates, or pick a major scene city. "
                "Amsterdam, Paris, Barcelona, Belgrade, and Zurich tend to have the most listings."
            )
        else:
            st.markdown(f"**{len(events)} rave parties in {res['city']}** via Resident Advisor")
            for evt in events[:30]:
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

    st.title("Berlin Rave Atlas")
    st.caption(
        "Your guide to Berlin's electronic music scene. Find the right night, learn "
        "the music, build a set, and browse raves across Europe."
    )

    tab_weekend, tab_mix, tab_learn, tab_beyond = st.tabs(
        ["Plan Your Night", "Rave Set Builder", "Learn the Scene", "Beyond Berlin"]
    )
    with tab_weekend:
        _tab_weekend(settings)
    with tab_mix:
        _tab_mix_builder(settings)
    with tab_learn:
        _tab_learn(settings)
    with tab_beyond:
        _tab_beyond_berlin(settings)


if __name__ == "__main__":
    main()
