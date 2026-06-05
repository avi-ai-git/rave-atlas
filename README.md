# 🎛️ Rave Atlas

> **Berlin's electronic music agent** — weekend event concierge, music education, and set-list builder in one Streamlit app.

[![Tests](https://img.shields.io/badge/tests-124%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-orange)](https://claude.ai/code)

---

## What is Rave Atlas?

Berlin has the densest electronic music scene on earth. Planning a night out means 20 open tabs, domain knowledge you may not have, and still picking wrong. Rave Atlas solves this with a single AI agent that:

- **Finds live Berlin events** from Resident Advisor and _reasons_ about which match your taste and budget — not just lists them
- **Teaches you the music** — genres, BPM signatures, scene history, record labels — from a curated knowledge base
- **Builds playable set lists** with a DJ energy arc, 30-second Deezer previews, and YouTube links
- **Learns your preferences** — rates events with 👍/👎, stores taste in SQLite, sends a Friday-morning weekend briefing

---

## Why an agent — not a chatbot or a RAG app

The task requires **runtime decisions**. When a user says "find hypnotic techno under €20 near Kreuzberg this Friday," the agent must:

1. Call `find_events` (RA GraphQL) for Berlin events this Friday
2. Optionally call `enrich_artist` (Discogs) for lineup context
3. Call `compare_events` (LLM with a reasoning rubric) to rank by fit
4. Synthesise a recommendation grounded in real data

A static chain decides the path at author-time. A ReAct agent decides at runtime. That branching is the distinction — and the reason a plain chatbot or retrieval-only app can't solve this problem.

---

## Features

| Tab | What it does |
|-----|-------------|
| 🗓 **This Weekend** | Live RA events · compare & rank by taste · thumbs rating loop · Friday digest |
| 📚 **Learn** | RAG over curated KB: genres, scene history, labels, DJ theory |
| 🎛 **Crate** | Set-list builder: energy arc + Deezer previews + YouTube links |

**Sidebar controls:** model picker (6 models, 2 providers) · tone radio · temperature + top-p sliders · per-query token/cost display · clear-chat

---

## Architecture

```
User prompt
    │
    ▼
safety.py ──── validate_input (length, duplicate guard)
    │           RateLimiter (20 req / 60 s per session)
    │           moderate() (Mistral classifier, score-gated — OWASP LLM01)
    │           fence() (wraps untrusted content as data, not commands)
    ▼
agent.py ───── LangGraph ReAct agent  (langchain.agents.create_agent)
    │           SqliteSaver checkpointer (cross-rerun conversation memory)
    │           SQLite taste profile (genre preferences · loved artists · budget)
    │
    ├── explain_music()   ── ChromaDB RAG, allowlist filter, gap-honesty
    ├── find_events()     ── Resident Advisor GraphQL, client-side filters
    ├── compare_events()  ── LLM with structured reasoning rubric
    ├── enrich_artist()   ── Discogs primary → MusicBrainz fallback
    └── build_setlist()   ── LLM energy arc + Deezer previews + YouTube URLs
```

**Storage:** ChromaDB vector store (local sentence-transformers, no API cost) + SQLite (conversations + taste profile + digests)

**Observability:** LangSmith traces every LLM call and tool invocation when `LANGCHAIN_TRACING_V2=true`

---

## How it was built — agentic coding methodology

This project was built in **15 phases** using [Claude Code](https://claude.ai/code) as the primary tool. Each phase had:

- A single deliverable (one module or concept)
- A model matched to the work: **Sonnet** for scaffolding and API wrappers, **Opus** for user-visible quality (prompts, set-list, agent orchestration)
- A thinking budget proportional to decision complexity (`think` → `think harder` → `ultrathink`)
- A test block that ran before the phase was committed

| Model allocation | Phases | Rationale |
|-----------------|--------|-----------|
| Claude Sonnet 4.6 | 0–4, 6–7, 9, 11–14 | Scaffolding, API wrappers, CRUD — one correct shape, no design ambiguity |
| Claude Opus 4.7 | 5 (prompts), 8 (setlist), 10 (agent) | User-visible quality + orchestration correctness directly affect output |

Sonnet built ~70% of the codebase. Opus was reserved for the three phases where reasoning quality determines output quality. The git log records which model built each component.

---

## Quick start — local

```bash
# 1. Clone
git clone <your-repo-url>
cd rave-atlas

# 2. Copy and fill secrets
cp .env.example .env
# Edit .env — at minimum: OPENROUTER_API_KEY, DISCOGS_TOKEN

# 3. Install (uv manages the virtualenv automatically)
uv sync

# 4. Seed the knowledge base (once, or after editing knowledge_base/*.md)
uv run python ingest.py

# 5. Run
uv run streamlit run app.py

# 6. Tests (all offline — no API calls)
uv run pytest
```

> **OpenRouter note:** If you see 404 errors on model calls, visit
> [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) and enable providers.
> The default model (Haiku 4.5) works without this change.

---

## Streamlit Cloud deployment

### One-time setup

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `master`, main file `app.py`
4. Open **Advanced settings → Secrets** and paste the contents of `.streamlit/secrets.toml.example` with your real values

### What happens on first deploy

The `data/` directory (ChromaDB + SQLite) is gitignored and starts empty on every fresh Streamlit Cloud container. On cold start, the app automatically:

1. Downloads the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB, cached by Streamlit's `@st.cache_resource`)
2. Embeds all seven knowledge-base markdown files into ChromaDB (~30 seconds)
3. Shows a spinner: _"Building knowledge base — first run only…"_

Subsequent restarts reuse the cached resource and start instantly.

### Minimum required secrets

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."   # required
DISCOGS_TOKEN      = "..."             # required for artist enrichment
```

All other secrets are optional (Mistral moderation, LangSmith, Ollama Cloud).

---

## Environment variables reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key (Claude models) |
| `DISCOGS_TOKEN` | ✅ | — | Discogs personal access token |
| `MISTRAL_API_KEY` | ✗ | — | Mistral moderation API (fails-open if absent) |
| `LANGSMITH_API_KEY` | ✗ | — | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | ✗ | `false` | Set to `true` to enable LangSmith |
| `OLLAMA_API_KEY` | ✗ | — | Ollama Cloud (open-source models in picker) |
| `DEFAULT_MODEL` | ✗ | `anthropic/claude-haiku-4.5` | Override the default model |
| `CHROMA_DIR` | ✗ | `data/chroma` | ChromaDB storage path |
| `SQLITE_PATH` | ✗ | `data/rave_atlas.db` | SQLite storage path |
| `MODERATION_THRESHOLD` | ✗ | `0.7` | Mistral score above which to block |
| `RATE_LIMIT_REQUESTS` | ✗ | `20` | Max requests per window per session |
| `RATE_LIMIT_WINDOW_SECONDS` | ✗ | `60` | Rate-limit window in seconds |

---

## Running tests

```bash
uv run pytest           # all 124 tests, ~7 seconds
uv run pytest -v        # verbose output
uv run pytest tests/test_injection.py   # OWASP injection corpus only
```

**Test coverage:**

| File | Tests | What's covered |
|------|-------|---------------|
| `test_safety.py` | 28 | `validate_input`, `RateLimiter`, `fence`, `moderate` |
| `test_injection.py` | 37 | 19 OWASP/jailbreak vectors · 12 false-positive music queries · fence structural tests |
| `test_tools.py` | 24 | `explain_music` (ChromaDB mock) · `enrich_artist` (Discogs/MusicBrainz mock) · `find_events` (RA mock) |
| `test_setlist.py` | 15 | `build_setlist` shape · energy clamping · Deezer hit/miss · LLM failure fallback |

All tests run offline — every external API is mocked. No network access required.

---

## Module map

```
rave-atlas/
├── app.py                  Thin Streamlit UI — tab routing and rendering only
├── agent.py                LangGraph ReAct agent + run_agent()
├── config.py               All settings from env vars; no hard-coded values
├── llm_client.py           OpenRouter + Ollama Cloud; cache, retry, cost
├── logging_config.py       structlog JSON to stdout
├── memory.py               SqliteSaver checkpointer + SQLite taste profile
├── safety.py               validate_input, moderate, RateLimiter, fence
├── ingest.py               Chunks KB markdown → ChromaDB (run once)
├── prompts/
│   ├── system.py           Agent persona + tool-routing instructions
│   ├── setlist.py          SETLIST_PROMPT with few-shot energy-arc examples
│   └── compare.py          Event-ranking reasoning rubric
├── tools/
│   ├── music_kb.py         explain_music: RAG, allowlist, gap-honesty
│   ├── events.py           find_events (RA GraphQL) + compare_events (LLM)
│   ├── artists.py          enrich_artist: Discogs → MusicBrainz fallback
│   └── setlist.py          build_setlist: LLM arc + Deezer + YouTube
├── automation/
│   └── weekend_digest.py   APScheduler Fri 09:00 → Fri–Tue briefing → SQLite
├── knowledge_base/         Curated markdown: genres, labels, history, theory
└── tests/                  124 offline pytest tests
    ├── conftest.py          autouse cache-clearing fixture
    ├── test_safety.py
    ├── test_injection.py
    ├── test_tools.py
    └── test_setlist.py
```

---

## Security

- **OWASP LLM01 (Prompt Injection):** Mistral moderation API with per-category score gating (not regex). Tested against 19 injection vectors.
- **Prompt fencing:** All untrusted input is wrapped in `=== BEGIN DATA ===` delimiters with a data-not-instructions directive before reaching the LLM.
- **Rate limiting:** 20 requests / 60 seconds per session, configurable.
- **Input validation:** Length bounds, duplicate-submission guard, whitespace normalisation.
- **Retrieval allowlists:** `explain_music` uses allowlists (not blacklists) for doc_type filtering.
- **Gap-honesty:** The agent surfaces when it doesn't know rather than hallucinating.

---

## Known limitations

| Limitation | Impact | Migration path |
|------------|--------|---------------|
| RA has no official public API | `find_events` uses an unofficial GraphQL endpoint — may change | Maintained third-party scraper or RA affiliate programme |
| Spotify audio-features API deprecated (Nov 2024) | No programmatic BPM/energy data | Deezer public API (used) + AcousticBrainz |
| SQLite is local-only | Memory doesn't persist across Streamlit Cloud container restarts | Firebase Realtime DB or PostgreSQL + pgvector |
| ChromaDB seeded at runtime | ~30-second cold start on Streamlit Cloud | Commit pre-built ChromaDB or use a hosted vector DB |
| Ollama Cloud model availability | `gemma3:27b` / `gpt-oss:120b` depend on Ollama Cloud's hosted catalogue | Fall back to OpenRouter equivalents |

---

## Prompt engineering vs RAG vs agent — where each is used

| Technique | Where | Why |
|-----------|-------|-----|
| **Prompt engineering** | `prompts/system.py`, `prompts/setlist.py`, `prompts/compare.py` | Shapes every LLM generation — persona, few-shot examples, event-ranking rubric. Applied at author-time to a single step. |
| **RAG** | `tools/music_kb.py → explain_music` | Grounds music-theory answers in the curated KB. Used when facts must be sourced, not invented. |
| **Agent (ReAct)** | `agent.py → run_agent()` | Used for the full application because the task requires multi-step reasoning with runtime tool selection — a decision that cannot be made at author-time. |

RAG alone powers only the Learn tab. Only the agent handles "find me hypnotic techno under €20 near Kreuzberg this Friday" — which needs event fetching, artist enrichment, taste-profile comparison, and answer synthesis in one turn.

---

*Built with [Claude Code](https://claude.ai/code) — Anthropic's CLI for agentic software engineering.*
