"""
Rave Atlas — Streamlit UI.

Four tabs, a Berlin-first agent, and one deliberate non-agent browser:

  🗓 Raves in Berlin  — Berlin event discovery, compare, ratings (ReAct agent)
  📖 Rave Wiki        — music education chat, grounded in the KB (ReAct agent, RAG)
  🎛 Set Builder      — 1-hour set builder (direct, deterministic tool call)
  🌍 Explore Europe   — browse other European Resident Advisor cities (plain API browse)

Why the split: the agent earns its cost where runtime reasoning matters
(planning a Berlin night across several tools). Mix Builder is a single focused
artefact, so it calls build_setlist directly — that guarantees every set is
fully enriched with Deezer previews + YouTube links, every time. Beyond Berlin
is pure retrieval, so it's an honest, fast browse rather than a chat.

Chat model: each tab keeps its own message list AND its own agent conversation
thread, so a Learn question never bleeds into the Weekend planner's context.
Messages render oldest→newest directly in the page; Streamlit's chat_input is
sticky at the bottom. On submit we append and rerun, so everything renders
through one path.

All agent/tool logic lives in agent.py and tools/. This file is presentation
and routing only — no LLM calls except the direct build_setlist used by Mix Builder.
"""

from __future__ import annotations

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
    padding-bottom: 5rem;   /* room for the sticky chat input */
    max-width: 860px;
}
h1 { letter-spacing: -0.03em; font-weight: 800; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0.2rem; border-bottom: 1px solid #222; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.45rem 1rem;
    font-weight: 600;
    color: #888;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(214,48,49,0.1);
    border-bottom: 2px solid #D63031;
    color: #F0F0F0 !important;
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
    font-size: 0.95rem;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton button { border-radius: 10px; }

/* ── Metrics & captions ──────────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.1rem; }
.stCaption, [data-testid="stCaptionContainer"] { opacity: 0.78; }

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
    .stTabs [data-baseweb="tab"] { padding: 0.4rem 0.6rem; font-size: 0.78rem; }
    h1 { font-size: 1.6rem; }
    .stChatInput > div { border-radius: 10px !important; }
}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Knowledge-base warm-up (Streamlit Cloud cold start) ───────────────────────

@st.cache_resource(show_spinner="Building knowledge base — first run only…")
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
    st.session_state.setdefault("mix_sets", [])            # list[{seed, setlist}]
    st.session_state.setdefault("intl_results", None)      # last Beyond Berlin search
    st.session_state.setdefault("total_cost", 0.0)
    st.session_state.setdefault("total_tokens", 0)
    st.session_state.setdefault("rated_events", set())


def _thread_key(tab: str) -> str:
    """Per-tab LangGraph thread so conversations don't cross-contaminate."""
    return f"{st.session_state['session_id']}:{tab}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🎛️ Rave Atlas")
        st.caption("Berlin's electronic music scene — planner, music school, mix builder.")
        st.divider()

        st.markdown("**Model & voice**")
        model_names = [m["name"] for m in config.AVAILABLE_MODELS]
        model_ids   = [m["id"]   for m in config.AVAILABLE_MODELS]
        default_idx = model_ids.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in model_ids else 0

        if len(config.AVAILABLE_MODELS) == 1:
            # Single-model mode: no picker, just show the model name as a label.
            model_id = model_ids[0]
            st.caption(f"Model: **{model_names[0]}**", help="To enable more models, uncomment entries in config.py AVAILABLE_MODELS.")
        else:
            selected_name = st.selectbox("Model", model_names, index=default_idx,
                                         help="Which LLM answers. Haiku is fastest/cheapest; Opus is highest quality.")
            model_id = model_ids[model_names.index(selected_name)]

        tone = st.radio(
            "Tone",
            options=["concise", "elaborated", "expert"],
            index=0,
            horizontal=True,
            help=(
                "concise = straight to the point. "
                "elaborated = full context and reasoning. "
                "expert = insider level — labels, BPMs, scene history."
            ),
        )

        with st.expander("Advanced — response sampling", expanded=False):
            st.caption(
                "Leave these as-is unless you want to experiment. Higher temperature "
                "= more varied/creative wording; lower = more focused and repeatable."
            )
            temperature = st.slider("Temperature (creativity)", 0.0, 1.5, 0.7, 0.05)
            top_p       = st.slider("Top-p (vocabulary breadth)", 0.5, 1.0, 1.0, 0.05)

        st.divider()
        st.caption("Session usage")
        col_tok, col_cost = st.columns(2)
        col_tok.metric("Tokens", f"{st.session_state['total_tokens']:,}")
        col_cost.metric("Cost",  f"${st.session_state['total_cost']:.4f}")

        if st.button("Clear chat history", use_container_width=True):
            for tab in ("weekend", "learn"):
                st.session_state[f"messages_{tab}"] = []
            st.session_state["mix_sets"]      = []
            st.session_state["total_cost"]    = 0.0
            st.session_state["total_tokens"]  = 0
            st.session_state["rated_events"]  = set()
            st.rerun()

        st.caption(f"Session `{st.session_state['session_id'][:8]}…`")

    return {"model_id": model_id, "tone": tone, "temperature": temperature, "top_p": top_p}


# ── Small shared renderers ────────────────────────────────────────────────────

def _render_tool_trace(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🔧 Tool trace — {len(tool_calls)} call(s)", expanded=False):
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
    """One-line 'date · time · venue (area)' for an event card."""
    bits: list[str] = []
    if evt.get("date"):
        bits.append(evt["date"])
    if evt.get("start_time"):
        bits.append(evt["start_time"])
    venue = evt.get("venue") or ""
    area = evt.get("area") or ""
    if venue:
        bits.append(f"{venue}" + (f" · {area}" if area else ""))
    return "  ·  ".join(bits)


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
    url  = evt.get("url") or ""
    with st.container(border=True):
        head_left, head_right = st.columns([8, 2])
        rank_badge = f'<span class="ra-rank">#{rank}</span> ' if rank else ""
        if url:
            head_left.markdown(f"{rank_badge}**[{name} ↗]({url})**", unsafe_allow_html=True)
        else:
            head_left.markdown(f"{rank_badge}**{name}**", unsafe_allow_html=True)

        # Price: always show something — never leave it blank
        price = (
            evt.get("price")
            or ("Free" if evt.get("price_numeric") == 0 else "No price listed")
        )
        head_right.markdown(f"**{price}**")

        meta = _event_meta_line(evt)
        if meta:
            st.caption(meta)
        if fit:
            st.markdown(fit)

        lineup = evt.get("lineup") or []
        genres = evt.get("genres") or []
        if genres:
            st.markdown(_chips(genres), unsafe_allow_html=True)
        if lineup:
            st.caption("Lineup: " + ", ".join(lineup[:8]) + ("…" if len(lineup) > 8 else ""))

        if rating_ctx is not None:
            _rating_buttons(evt, name, rank, *rating_ctx)


def _rating_buttons(evt: dict, name: str, rank: int | None, session_id: str, msg_idx: int) -> None:
    rated = st.session_state["rated_events"]
    rating_key = f"r_{msg_idx}_{name}"
    if rating_key in rated:
        st.caption("rated ✓")
        return
    c1, c2, _ = st.columns([1, 1, 6])
    if c1.button("👍", key=f"up_{msg_idx}_{name}", help="Good fit — show me more like this"):
        memory.update_profile_from_feedback(session_id, evt, liked=True)
        rated.add(rating_key)
        st.toast("Liked — profile updated", icon="✅")
        st.rerun()
    if c2.button("👎", key=f"dn_{msg_idx}_{name}", help="Not for me"):
        memory.update_profile_from_feedback(session_id, evt, liked=False)
        rated.add(rating_key)
        st.toast("Noted — will avoid similar events", icon="🚫")
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
        return  # find_events wasn't called on this turn
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
            rating_ctx=(session_id, msg_idx) if ranked_list else None,
        )


# ── Set-list card (Set Builder) ───────────────────────────────────────────────

def _energy_bar(energy: int) -> str:
    """Render an energy value as a mini visual bar (e.g. ████░░ 6/10)."""
    filled = round(energy / 2)   # out of 5 blocks for compactness
    empty  = 5 - filled
    return "█" * filled + "░" * empty + f" {energy}/10"


def _render_setlist(sl: dict) -> None:
    tracks = sl.get("tracks", [])
    if not tracks:
        st.warning("Couldn't build a playable set just now — the music API or LLM may be busy. Try again.")
        return

    st.markdown(f"### {sl.get('title', 'Set')}")
    arc = sl.get("energy_arc", [])
    if arc:
        st.caption(f"Arc: **{' → '.join(str(e) for e in arc)}**   ({len(tracks)} tracks, ~{len(tracks) * 4} min)")

    # ── Summary table (quick overview at a glance) ────────────────────────────
    with st.expander("📋 Track list at a glance", expanded=True):
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
            c_info.markdown(f"**{t.get('artist', '')}** — *{t.get('title', '')}*")
            c_nrg.markdown(_energy_bar(t.get("energy", 5)))

            if t.get("reason"):
                st.caption(t["reason"])

            links = []
            if t.get("deezer_url"):
                links.append(f"[Deezer ↗]({t['deezer_url']})")
            if t.get("youtube_url"):
                links.append(f"[YouTube ↗]({t['youtube_url']})")
            if links:
                st.caption(" · ".join(links))

            if t.get("preview_url"):
                # Label clearly: is this the exact track or an artist fallback?
                if t.get("deezer_fallback"):
                    st.caption(
                        f"🎵 30-sec preview: similar track by **{t.get('artist', '')}** "
                        f"(exact title not found on Deezer)"
                    )
                else:
                    st.caption("🎵 30-sec preview")
                st.audio(t["preview_url"], format="audio/mp3")


# ── Chat message renderer (Weekend / Learn) ───────────────────────────────────

def _render_chat(tab_key: str, empty_hint: str) -> None:
    messages = st.session_state[f"messages_{tab_key}"]
    if not messages:
        st.caption(empty_hint)
        return
    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.warning(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg["role"] == "assistant" and not msg.get("blocked"):
                tc = msg.get("tool_calls", [])
                if tab_key == "weekend":
                    _render_events_block(tc, idx)
                _render_tool_trace(tc)
                if msg.get("usage"):
                    _render_cost_badge(msg.get("model", ""), msg["usage"], msg.get("cost_estimate", 0.0))


def _handle_chat_input(tab_key: str, settings: dict, placeholder: str) -> None:
    prompt = st.chat_input(placeholder, key=f"chat_input_{tab_key}")
    if not prompt:
        return
    messages = st.session_state[f"messages_{tab_key}"]
    messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking…"):
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
    st.session_state["total_cost"]   += result["cost_estimate"]
    st.session_state["total_tokens"] += result["usage"]["total_tokens"]
    st.rerun()


# ── Digest section (Berlin only) ──────────────────────────────────────────────

def _render_digest_section() -> None:
    session_id = st.session_state["session_id"]
    digest = memory.load_digest(session_id) or memory.load_digest("__global__")
    with st.expander("📋 Berlin weekend digest" + ("" if digest else " — tap to generate"), expanded=False):
        if digest:
            st.markdown(digest)
        else:
            st.info(
                "No digest yet. In production a Friday-morning GitHub Actions job sends "
                "this to Telegram; here you can generate it on demand."
            )
        if st.button("Generate / refresh digest", key="btn_regen_digest"):
            with st.spinner("Fetching Berlin events and writing the digest…"):
                new_digest = generate_digest(session_id)
            if new_digest:
                st.success("Digest ready.")
                st.rerun()
            else:
                st.warning("Could not generate the digest — Resident Advisor or the LLM may be busy.")


# ── Tabs ──────────────────────────────────────────────────────────────────────


def _tab_weekend(settings: dict) -> None:
    st.markdown(
        "**Your Berlin event agent.** Ask what's on any weekend, compare venues, "
        "or describe your taste and get ranked picks. Every event name is a direct "
        "link to its Resident Advisor page. Rate picks 👍/👎 to teach the agent your taste."
    )
    st.caption(
        "Not limited to 'this weekend' — ask about next Friday, any specific date, "
        "or describe a vibe and let it find the right night."
    )
    st.divider()
    _render_digest_section()
    _render_chat("weekend", "Ask about Berlin events to get started — any date, any genre, any price range.")
    _handle_chat_input("weekend", settings,
                       placeholder="What's on this Friday? / Hypnotic techno under €20 / Which fits my taste?")


def _tab_learn(settings: dict) -> None:
    st.markdown(
        "**Your electronic music and rave culture reference.** Genres and subgenres, "
        "BPM signatures, harmonic analysis, Berlin scene history, record labels, what to "
        "expect at your first rave, and how to look after yourself after. "
        "Answers come from the curated Rave Atlas knowledge base."
    )
    st.caption(
        "Try: genre history, BPM ranges, what a specific label sounds like, "
        "rave etiquette and door culture, harm reduction, chord theory in house vs techno."
    )
    st.divider()
    _render_chat("learn", "Ask anything about electronic music, Berlin's scene, or rave culture.")
    _handle_chat_input("learn", settings,
                       placeholder="What is minimal techno? / History of Tresor / What should I bring to a rave?")


def _tab_mix_builder(settings: dict) -> None:
    st.markdown(
        "**Build a rave setlist from a brief.** Describe a slot — time of night, "
        "venue vibe, BPM target, energy level — and get a full 1-hour set back: "
        "16 tracks, a deliberate energy arc from warm-up to peak to close, "
        "30-second Deezer previews, and YouTube links for every track."
    )
    st.caption(
        "The arc is the point — not just a list of tracks, but a journey with warm-up, "
        "build, peak, and comedown phases. Be specific: 'hypnotic 130bpm 2am Berghain' "
        "beats 'techno.'"
    )
    st.divider()
    sets = st.session_state["mix_sets"]
    if not sets:
        st.info("Describe a slot or vibe below and I'll build a playable 1-hour set.")
    for entry in sets:
        with st.chat_message("user"):
            st.markdown(entry["seed"])
        with st.chat_message("assistant"):
            _render_setlist(entry["setlist"])

    seed = st.chat_input(
        "Warm-up for KitKat Saturday 23h / Hypnotic 2am Berghain set / Sunday Sisyphos comedown",
        key="chat_input_mix",
    )
    if seed:
        with st.spinner("Building the set and fetching Deezer previews… (~30 s for 16 tracks)"):
            sl = build_setlist(seed=seed, n=16)
        st.session_state["mix_sets"].append({"seed": seed, "setlist": sl})
        st.rerun()


def _tab_beyond_berlin(settings: dict) -> None:
    st.markdown(
        "**Live Resident Advisor listings for any European city.** "
        "The Berlin agent has full local depth, but the RA events tool works across Europe. "
        "Pick a city, set your dates, and get real listings directly from Resident Advisor."
    )
    st.caption(
        "Coverage is best for major scenes: Amsterdam, Paris, Barcelona, Belgrade, "
        "Vienna, Zurich, Copenhagen. Smaller cities may return few or no results. "
        "This is a direct browse — not an agent conversation."
    )
    st.divider()

    col_city, col_from, col_to = st.columns([2, 1, 1])
    # Show all cities except Berlin (home city already covered in This Weekend)
    other_cities = [c for c in config.AVAILABLE_CITIES if c != "Berlin"]
    city = col_city.selectbox("City", other_cities, index=0)

    today = date.today()
    default_from = today + timedelta(days=(4 - today.weekday()) % 7)  # next/this Friday
    default_to = default_from + timedelta(days=3)
    date_from = col_from.date_input("From", value=default_from)
    date_to = col_to.date_input("To", value=default_to)

    col_genre, col_price = st.columns([3, 2])
    genres = col_genre.multiselect(
        "Genres (optional)",
        ["Techno", "House", "Minimal", "Tech House", "Trance", "Drum & Bass",
         "Dubstep", "Ambient", "Disco", "Electro", "Hardcore", "Industrial", "EBM"],
        help="Filters the returned listings by RA's own genre tags.",
    )
    max_price = col_price.slider("Max price (€, 0 = no limit)", 0, 80, 0, 5)

    if st.button("Find events on RA", type="primary"):
        if date_to < date_from:
            st.error("'To' date must be on or after 'From' date.")
        else:
            filters: dict[str, Any] = {}
            if genres:
                filters["genres"] = genres
            if max_price > 0:
                filters["max_price"] = float(max_price)
            from tools.events import find_events as _find
            with st.spinner(f"Querying Resident Advisor for {city}…"):
                events = _find(date_from.isoformat(), date_to.isoformat(), filters=filters or None, city=city)
            st.session_state["intl_results"] = {"city": city, "events": events}
            st.rerun()

    res = st.session_state.get("intl_results")
    if res:
        events = res["events"]
        st.divider()
        if not events:
            st.info(
                f"No Resident Advisor listings found for **{res['city']}** in that date range. "
                "RA's coverage varies — larger cities (Amsterdam, Paris, Barcelona, Vienna, "
                "Copenhagen, Belgrade) return the most results. Try widening the dates, "
                "or pick a different city."
            )
        else:
            st.markdown(f"**{len(events)} event(s) in {res['city']}** via Resident Advisor")
            for evt in events[:30]:
                _render_event_card(evt)


# ── Onboarding ────────────────────────────────────────────────────────────────

def _render_intro() -> None:
    no_activity = (
        not st.session_state["messages_weekend"]
        and not st.session_state["messages_learn"]
        and not st.session_state["mix_sets"]
    )
    with st.expander("👋 New here? How Rave Atlas works", expanded=no_activity):
        st.markdown(
            "**Rave Atlas is Berlin-first.** The agent knows the venues, labels, and scene "
            "history in real depth. Everything else is covered honestly but without the same detail.\n\n"
            "- **🗓 Raves in Berlin** — live Berlin events from Resident Advisor, ranked by "
            "fit to your taste. Any date, any genre, any price range. Every event name links "
            "straight to its RA page. Rate picks 👍/👎 to teach the agent your preferences.\n"
            "- **📖 Rave Wiki** — genres, BPM signatures, scene history, labels, track "
            "anatomy, rave culture, harm reduction. Grounded in the curated knowledge base.\n"
            "- **🎛 Set Builder** — describe a slot or vibe, get a 1-hour playable set: "
            "16 tracks with a warm-up / build / peak / close arc, Deezer previews, and YouTube links.\n"
            "- **🌍 Explore Europe** — direct RA event browse for any European city. "
            "Coverage is best for major scenes (Amsterdam, Paris, Barcelona, Belgrade, Zurich, Copenhagen).\n\n"
            "Adjust tone in the sidebar: **concise** (just the facts), **elaborated** (full context), "
            "or **expert** (insider depth with labels, BPMs, and scene history)."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Rave Atlas", page_icon="🎛️", layout="wide",
                       initial_sidebar_state="expanded")
    _inject_css()
    _ensure_kb_seeded()
    _init_session()
    get_scheduler()  # local convenience; production automation runs via GitHub Actions

    settings = _render_sidebar()

    st.title("🎛️ Rave Atlas")
    st.caption("Berlin's electronic music scene — planner, music school, mix builder.")
    _render_intro()

    tab_weekend, tab_learn, tab_mix, tab_beyond = st.tabs(
        ["🗓 Raves in Berlin", "📖 Rave Wiki", "🎛 Set Builder", "🌍 Explore Europe"]
    )
    with tab_weekend:
        _tab_weekend(settings)
    with tab_learn:
        _tab_learn(settings)
    with tab_mix:
        _tab_mix_builder(settings)
    with tab_beyond:
        _tab_beyond_berlin(settings)


if __name__ == "__main__":
    main()
