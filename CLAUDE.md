# CLAUDE.md — Rave Atlas

> **Dual-purpose document.**
> (1) Onboarding context for Claude Code in future sessions — read this before touching any file.
> (2) Process document for reviewers and hiring managers — shows how this project was designed and built using agentic coding methodology.

---

## What is Rave Atlas?

### The problem

Berlin has the densest electronic music scene on earth. Planning a night out is fragmented and high-effort: events are scattered across Resident Advisor, Instagram, and word of mouth; you can't easily tell which party matches your taste without domain knowledge; and newcomers and tourists don't know the genres, the labels, or which artists are worth the trip. You waste a Friday afternoon with twenty tabs open and still pick wrong.

### The solution

A single AI agent that:
- Fetches live Berlin events from Resident Advisor and **reasons** about which match your taste and budget — not just lists them
- Teaches you the music — genres, history, Berlin's labels (Tresor, Ostgut Ton / Berghain, BPitch) — from a curated knowledge base
- Builds you a set list with an energy arc and **playable** 30-second Deezer previews + YouTube links
- Remembers your preferences across sessions and sends a Friday-morning briefing covering the weekend through Tuesday

### Why an agent (not a chatbot or a RAG app)

The task requires **runtime decisions**: fetch events, or enrich an artist, or retrieve music theory, or build a set — chosen dynamically based on what the user asks. A static chain decides the path at author-time; an agent decides at runtime. That branching logic is exactly what a ReAct agent does, and it's the reason a plain chatbot or a retrieval-only app can't solve this problem. See §Architecture decisions for the full reasoning.

### Target users

1. Berlin locals who go out weekly and want better-matched nights with less research effort
2. Newcomers and tourists who love electronic music but don't know the scene
3. Curious learners who want music education (genre history, theory, labels) without a formal course

---

## How this was built — agentic coding methodology

This project was designed and built phase-by-phase using **Claude Code** (Anthropic's CLI) as the primary agentic coding tool. The methodology is not "ask the AI to write everything at once" — it is a structured 15-phase process where each phase had:

- A **specific deliverable** (one module or concept)
- A **chosen model** matched to what that phase required
- A **reasoning budget** (thinking level) proportional to the decision complexity
- A **test block** that ran before committing — no phase was merged until it passed

### The model strategy

| Work type | Model | Thinking | Rationale |
|---|---|---|---|
| Scaffolding, API wrappers, CRUD plumbing | Claude Sonnet 4.6 | Off | One correct shape; no design decisions; spend nothing |
| Schema design, retrieval logic, UI state flow | Sonnet | `think harder` | Hard to change once downstream code depends on it |
| User-visible output quality: prompts, set-list | Claude Opus 4.7 | `think harder` | This is what reviewers read and users see; Opus earns its cost |
| Agent orchestration (agent.py — the heart) | Claude Opus 4.7 | `ultrathink` | One wrong import or loop design breaks all 14 other phases |

This is deliberate allocation, not "use the best model for everything." Sonnet built ~70% of the codebase; Opus was reserved for the three phases where reasoning quality directly determines output quality. The git history records which model built each component.

### The 15-phase build sequence

Each phase produced one commit. The sequence is dependency-correct: each phase only uses files from earlier phases.

| Phase | Files produced | What it does | Model | Thinking |
|---|---|---|---|---|
| 0 | `pyproject.toml`, `config.py`, `logging_config.py`, `app.py`, `CLAUDE.md` | Scaffold + config + logging + this doc | Sonnet | Off |
| 1 | `knowledge_base/*.md`, `ingest.py` | 7 KB markdown files + ChromaDB ingestion pipeline | Sonnet | think |
| 2 | `tools/music_kb.py` | RAG retrieval with per-tool allowlist + gap-honesty | Sonnet | think |
| 3 | `safety.py` | Input validation + Mistral moderation + rate limiter + prompt fencing | Sonnet | think |
| 4 | `llm_client.py` | OpenRouter + Ollama Cloud client, cache, retry, cost estimation | Sonnet | think |
| 5 | `prompts/system.py`, `prompts/setlist.py`, `prompts/compare.py` | Agent persona, few-shot set-list examples, event-comparison reasoning rubric | **Opus** | think harder |
| 6 | `tools/artists.py` | `enrich_artist`: Discogs primary + MusicBrainz fallback | Sonnet | Off |
| 7 | `tools/events.py` | `find_events` (Resident Advisor GraphQL) + `compare_events` (LLM-ranked) | Sonnet | think |
| 8 | `tools/setlist.py` | `build_setlist`: energy-arc tracklist + Deezer 30s previews + YouTube links | **Opus** | think harder |
| 9 | `memory.py` | LangGraph `SqliteSaver` + SQLite taste profile + feedback loop | Sonnet | think harder |
| 10 | `agent.py` | LangGraph ReAct agent: wires all tools, memory, safety, prompts | **Opus** | ultrathink |
| 11 | `automation/weekend_digest.py` | APScheduler job: Friday AM → Fri–Tue briefing → written to store | Sonnet | think |
| 12 | `app.py` (full) | Streamlit UI: 3 tabs, model picker, sliders, tool-trace expander, ratings, cost display | Sonnet | think harder |
| 13 | `tests/` | pytest suite: safety, tools, injection corpus (OWASP + jailbreaks), setlist | Sonnet | think |
| 14 | `README.md`, LangSmith wiring | Observability, docs, Streamlit Cloud deploy prep | Sonnet | think |

---

## Architecture decisions and WHY

Each decision maps to a rubric criterion or reviewer fix. Nothing is arbitrary.

### 1. ReAct agent over static chain or plain RAG

**Decision:** LangGraph ReAct agent with runtime tool selection.

**Why:** The application has three fundamentally different task types (event lookup, music education, set building) plus cross-cutting enrichment (`enrich_artist`). A static chain hard-codes the path at author-time. A ReAct agent reads the user's message and decides at runtime which tool(s) to call, in what order, and when it has enough information to answer. This is the architectural distinction between an *agent* and a *chatbot* — exactly the concept the Sprint 3 rubric tests.

**Rubric mapping:** "Understanding core concepts — the learner can mention differences between different agent types" + "can explain function calling implementation clearly."

### 2. LangGraph `SqliteSaver` for long-term memory

**Decision:** LangGraph's built-in `SqliteSaver` checkpointer backed by SQLite, plus a separate taste-profile table in the same DB.

**Why:** Streamlit reruns the entire script on every interaction, so in-memory state resets constantly. `SqliteSaver` persists conversation threads to disk keyed by `session_id` — the page can reload and the agent picks up exactly where it left off. The taste profile (preferred genres, loved/blocked artists, budget ceiling) is a separate concern: it's not a conversation history, it's a user model that improves with feedback. Two tables, one file, zero extra infra.

**Rubric mapping:** Medium optional task #3 (long-term memory) + Hard optional task #4 (learns from feedback).

### 3. ChromaDB + local sentence-transformers (no API cost for embeddings)

**Decision:** `all-MiniLM-L6-v2` model runs locally; ChromaDB stores vectors on disk.

**Why:** Embedding the knowledge base with an API (OpenAI, Cohere) adds cost, a network dependency, and a rate limit. The local model is free, runs offline, and is fast enough for a 7-file KB. Vectors persist across restarts in `data/chroma/`.

### 4. Two LLM providers, five models, both OpenAI-API-compatible

**Decision:** OpenRouter for Anthropic Claude (Sonnet, Haiku, Opus); Ollama Cloud for open-source (Gemma 3 27B, GPT-OSS 120B).

**Why:** The cheatsheet and brief both require "the OpenAI API." OpenRouter is fully OpenAI-SDK-compatible (same client, same parameters) and routes to Anthropic behind one key — this satisfies the requirement while actually using Claude. Ollama Cloud (also OpenAI-compatible) adds open-source options via the same SDK call. `llm_client.py` routes on `model["provider"]` to select the right base URL and key. Five models is a principled, curated list — not a dropdown of 100 options — which is easier to explain and defend.

**Rubric mapping:** "OpenAI API" requirement + Medium optional task #9 (multi-model support).

### 5. Mistral moderation over regex for injection detection

**Decision:** Call the Mistral moderation API, gate on per-category probability scores.

**Why:** A regex-based denylist (matching "ignore previous instructions") is easily bypassed — obfuscation, zero-width characters, paraphrasing. A classifier assigns probabilities across harm categories (violence, self-harm, illegal activity, etc.) regardless of surface form. Score-based gating is the OWASP LLM01 mitigation; substring matching is explicitly called out as insufficient in security literature. This was a Sprint 1 deduction; it won't be here.

**Rubric mapping:** "Appropriate security considerations" (technical implementation criterion) + OWASP LLM01.

### 6. Per-tool allowlists, not blacklists, for retrieval scope

**Decision:** `explain_music` accepts an `allowed_doc_types` list; if provided, only chunks with matching metadata are retrieved.

**Why:** Blacklists enumerate what to exclude — they miss anything not anticipated. Allowlists enumerate what to include — they're closed by default. For a tool that should only answer from the music KB (not from user-injected content or stray chunks), an allowlist is the correct model. This was a Sprint 2 deduction.

### 7. Gap-honesty: the agent says when it doesn't know

**Decision:** `explain_music` returns `grounded: False` when similarity is below threshold. `find_events` returns an empty list with a logged status when RA's endpoint fails. The agent's system prompt instructs it to surface these signals rather than invent an answer.

**Why:** An agent that fabricates events or invents music facts is worse than useless — it's misleading. Explicit gap-honesty (saying "no events found" or "this isn't in my knowledge base") is both the correct engineering choice and the honest-limitations style that both previous reviewers praised.

### 8. uv + pyproject.toml + committed lockfile

**Decision:** `uv` manages dependencies; `pyproject.toml` has sensible compatible-range pins; `uv.lock` is committed.

**Why:** Lower-bound-only pins (`requests>=2.0`) allow any later version, including breaking ones. The lockfile captures exact versions that were tested, making the build reproducible. `uv` is faster than pip, handles virtual environments automatically, and the `uv run` command means contributors don't need to manually activate an environment. This was a Sprint 1 deduction.

### 9. Modular structure: no monolithic `app.py`

**Decision:** Separate modules for each concern — `config.py`, `logging_config.py`, `safety.py`, `llm_client.py`, `memory.py`, `agent.py`, `prompts/`, `tools/`, `automation/`. `app.py` is thin UI routing only.

**Why:** A top-to-bottom `app.py` with 800 lines is untestable and unmaintainable. Each module here is independently importable and has a `if __name__ == "__main__"` test block. The test suite (`tests/`) can mock individual modules without touching the UI. This was deducted in both Sprint 1 and Sprint 2.

### 10. LangSmith for observability

**Decision:** LangSmith traces all LLM and tool calls when `LANGCHAIN_TRACING_V2=true`.

**Why:** The `.env` already contained LangSmith keys. It integrates natively with LangChain/LangGraph (zero extra code — just environment variables). Every agent run appears in the LangSmith dashboard with token counts, latency, and tool traces — satisfying the "metrics/observability" requirement that was deducted in Sprint 2.

---

## Reviewer fix-list — Sprint 1 + 2 deductions and their exact mitigations

| Deduction (sprint) | Mitigation | Where |
|---|---|---|
| No rate limiting (S1) | `RateLimiter` class with configurable window + count | `safety.py` |
| Regex-only injection filter (S1) | Mistral moderation API, score-based gating | `safety.py` |
| No tests (S1) | `pytest` parametrized suite, all offline | `tests/` |
| Monolithic `app.py` (S1) | `main()` entry point + small helpers; all logic in separate modules | `app.py`, all modules |
| No logging (S1) | structlog JSON to stdout on every call | `logging_config.py`, all modules |
| Lower-bound-only pins (S1) | Compatible-range pins + committed `uv.lock` | `pyproject.toml` |
| Sparse commits (S2) | One commit per phase, from the first line of code | git history |
| KB hard-coded in repo (S2) | ChromaDB vector store + SQLite for profile/sessions/digests | `data/` |
| No observability/metrics (S2) | LangSmith tracing (tokens, latency, tool traces) | `llm_client.py`, `agent.py` |
| Hard-coded models/URLs (S2) | Everything in `config.py` loaded from `.env` | `config.py` |
| Retrieval used blacklists (S2) | Per-tool `allowed_doc_types` allowlists | `tools/music_kb.py` |
| No injection test suite (S2) | OWASP corpus + jailbreaks + false-positive checks | `tests/test_injection.py` |
| No caching (S2) | In-memory cache keyed on (model, messages hash) | `llm_client.py` |
| Wrong `create_react_agent` import (S2) | `from langchain.agents import create_react_agent` | `agent.py` |
| Incomplete type hints (S2) | Full type hints on every function signature | all modules |

---

## Optional tasks implemented

**Hard tasks (target: 1 primary + extras):**
- ✅ **#1 Agentic RAG** — `explain_music` is invoked by the agent at runtime, not statically. The agent decides when to retrieve. *(primary hard)*
- ✅ **#2 LLM observability** — LangSmith dashboard for latency, token usage, model distribution
- ✅ **#4 Learns from user feedback** — thumbs-up/down updates the SQLite taste profile; future recommendations improve
- ✅ **#7 Deploy to cloud** — Streamlit Community Cloud

**Medium tasks (target: 2 primary + extras):**
- ✅ **#3 Long-term memory** — `SqliteSaver` persists conversations + SQLite taste profile across sessions *(primary medium)*
- ✅ **#4 External-API tool** — five external integrations: RA GraphQL, Discogs, MusicBrainz, Deezer, Ollama Cloud *(primary medium)*
- ✅ **#1 Token usage and cost display** — per-query tokens + cost estimate in Streamlit sidebar
- ✅ **#2 Retry logic** — exponential-backoff retry on transient LLM/API errors
- ✅ **#6 Caching** — in-memory cache on repeated LLM calls
- ✅ **#7 Feedback loop** — user ratings stored; taste profile updated
- ✅ **#9 Multi-model support** — OpenRouter (Anthropic) + Ollama Cloud (Gemma 3, GPT-OSS); model picker in UI

**Easy tasks:**
- ✅ **#2 Agent personality** — configurable tone (friendly / concise / formal) passed into system prompt
- ✅ **#3 Model picker** — sidebar dropdown from `AVAILABLE_MODELS`
- ✅ **#4 Temperature + top-p sliders** — sidebar controls

---

## Module map

```
rave-atlas/
├── app.py                  Thin Streamlit UI; main() + tab routing only
├── agent.py                LangGraph ReAct agent assembly + run_agent()
├── config.py               All env-var settings; no hard-coded values
├── llm_client.py           OpenRouter + Ollama Cloud client; cache, retry, cost
├── logging_config.py       structlog JSON setup; get_logger(name) helper
├── memory.py               SqliteSaver checkpointer + SQLite taste profile
├── safety.py               validate_input, moderate, RateLimiter, fence
├── ingest.py               Chunks KB markdown → ChromaDB (run once / on update)
├── prompts/
│   ├── system.py           Agent system prompt (persona, tool routing, tone)
│   ├── setlist.py          SETLIST_PROMPT with 2–3 few-shot energy-arc examples
│   └── compare.py          COMPARE_PROMPT — event-ranking reasoning rubric
├── tools/
│   ├── music_kb.py         explain_music (RAG, allowlist, gap-honesty)
│   ├── events.py           find_events (RA GraphQL) + compare_events (LLM)
│   ├── artists.py          enrich_artist (Discogs → MusicBrainz fallback)
│   └── setlist.py          build_setlist (LLM energy arc + Deezer + YouTube)
├── automation/
│   └── weekend_digest.py   APScheduler Fri-AM job → Fri–Tue briefing → SQLite
├── knowledge_base/         Curated markdown: genres, history, labels, theory
│   ├── genres_techno.md
│   ├── genres_house.md
│   ├── genres_psytrance.md
│   ├── genres_dubstep.md
│   ├── berlin_scene_history.md
│   ├── labels.md
│   └── track_anatomy.md
├── data/                   Gitignored: chroma/, rave_atlas.db
└── tests/
    ├── test_safety.py      Validation, rate limit, moderation (mocked)
    ├── test_tools.py       explain_music, enrich_artist, find_events (mocked APIs)
    ├── test_injection.py   OWASP corpus + jailbreaks + false-positive checks
    └── test_setlist.py     build_setlist output shape (mocked LLM)
```

---

## Running locally

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd rave-atlas

# 2. Copy secrets
cp .env.example .env
# Edit .env — fill in OPENROUTER_API_KEY, OLLAMA_API_KEY, DISCOGS_TOKEN,
#              MISTRAL_API_KEY, LANGSMITH_API_KEY

# 3. Install dependencies (uv manages the virtualenv automatically)
uv sync

# 4. Build the knowledge base (run once, or after editing knowledge_base/*.md)
uv run python ingest.py

# 5. Launch the app
uv run streamlit run app.py

# 6. Run the test suite
uv run pytest
```

---

## When this project uses prompt engineering vs RAG vs an agent

This is an explicit Sprint 3 rubric question. The project uses all three, deliberately layered:

| Technique | Where used | Why |
|---|---|---|
| **Prompt engineering** | `prompts/system.py`, `prompts/setlist.py`, `prompts/compare.py` | Shapes every LLM response — the agent's persona, its few-shot set-list examples, and the structured reasoning rubric for comparing events. Applied at author-time to a single generation step. |
| **RAG** | `tools/music_kb.py` → `explain_music` | Grounds music-theory answers in the curated KB. Used when facts must be sourced (not invented) and the answer depends on a static, curated corpus. |
| **Agent (ReAct)** | `agent.py` → `run_agent()` | Used for the full application because the task requires multi-step reasoning with runtime tool selection. The agent decides which of the five tools to call based on the user's request — a decision that cannot be made at author-time. |

RAG alone could power the Learn tab. Prompt engineering alone could build a fixed chatbot. Only the agent can handle "find me hypnotic techno this Friday under €20 near Kreuzberg" — which requires fetching events, enriching artists, comparing against the taste profile, and synthesising a recommendation in one turn.

---

## Known limitations and migration paths

| Limitation | Impact | Migration path |
|---|---|---|
| Resident Advisor has no official public API | `find_events` calls an unofficial GraphQL endpoint — it may change or break | Use a maintained third-party scraper (e.g., Apify's RA template) or the official RA affiliate programme if/when available |
| Spotify audio-features API deprecated (Nov 2024) | Cannot pull BPM/energy/danceability programmatically | Deezer's public API still works; AcousticBrainz is an open alternative |
| SQLite for memory is local-only | Won't work on stateless cloud deployments without a writable volume | Migrate to Firebase Realtime DB (direct drop-in for the profile table) or PostgreSQL + pgvector (replaces both SQLite and ChromaDB) |
| Knowledge base is static markdown | KB must be manually updated as genres evolve | Add an admin UI for KB editing + re-ingestion trigger |
| Ollama Cloud model availability | `gemma3:27b` and `gpt-oss:120b` availability depends on Ollama Cloud's hosted catalogue | Fall back to OpenRouter equivalents if Ollama Cloud changes its model roster |
