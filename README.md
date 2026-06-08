# Berlin Rave Atlas

> An AI agent for Berlin's electronic music scene: a weekend event concierge, a music teacher, a set builder, and a Europe-wide rave browser, in one app.

[![Tests](https://img.shields.io/badge/tests-256%20passing-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)](app.py)
[![Live demo](https://img.shields.io/badge/live-berlin--rave--atlas.streamlit.app-D63031)](https://berlin-rave-atlas.streamlit.app)

---

## The problem

Berlin has the densest electronic music scene on earth, and planning a night in it is fragmented, high-effort work. Events are scattered across Resident Advisor, Instagram, and word of mouth. You cannot tell which party matches your taste without already knowing the scene, and you can lose a Friday afternoon to twenty browser tabs and still pick wrong.

## Who it is for

**Berlin regulars** who go out every weekend but still spend Friday afternoon unsure which party matches their sound.

**First-timers and visitors** who love the music but do not know Berghain from about blank, or techno from deep house.

**Curious listeners** who want to understand the music, the genre history, the BPM signatures, and how a DJ set is shaped.

---

## What it does

Four things, each with its own tab.

**Your Berlin Guide** is the agent you talk to. Ask what's on this Friday, which night matches your taste, what Berghain's door is actually like, what minimal techno sounds like, the history of Tresor, or how to stay safe on a long night. It fetches live Resident Advisor events, reasons over lineups against your taste profile, pulls artist and label context from the knowledge base, and searches the web when needed. Parties, scene history, genre theory, harm reduction, all in one conversation. It learns from the events you rate.

**Berlin Raves** shows you what's on without a conversation. An AI picks digest at the top surfaces the standout nights for the week. Below that, a raw Resident Advisor browse lets you filter by date, price, and count. No agent, just the listings when you already know what you want.

**Rave Set Builder** builds a tracklist with a deliberate energy arc. Give it a vibe, a time of night, a track count, or a venue, and it returns each track with a function role (opener, build, peak, sustain, resolution, closer), a BPM target, a one-line reason for each position, and a 30-second Deezer preview plus a YouTube link. A Last.fm genre guard rejects non-electronic artists before the catalogue call so the set stays on genre. A two-to-three sentence set story describes the energy narrative across the whole arc. Tracks are grounded in Deezer's real catalogue so the titles are always playable, and the sidebar model selection applies here too.

**Raves Beyond Berlin** is a direct browse of Resident Advisor events for any European city, by date, genre, price, and area. No agent, just the listings, honestly.

---

## When it uses prompt engineering, retrieval, or an agent

The architecture is the choice of which technique fits each surface, not a single default.

| Technique | Where | Why |
|---|---|---|
| Prompt engineering | `prompts/` | Shapes every generation: persona, few-shot energy-arc examples, the ranking guide. Author-time, single step. |
| Retrieval (RAG) | Your Berlin Guide | Grounds answers in the curated KB. Used when facts must be sourced, not invented. |
| Agent (ReAct) | Your Berlin Guide | Multi-step reasoning with runtime tool selection. |
| Direct tool call | Berlin Raves, Rave Set Builder, Raves Beyond Berlin | One well-defined job where a deterministic call beats asking a model to decide to make it. |

**Why a ReAct agent and not a static chain or plain RAG.** The app has three task shapes plus cross-cutting enrichment. A chain fixes the tool path when the code is written; a retrieval-only app can only answer from documents. The agent reads the message and decides at runtime which tools to call, in what order, and when it has enough to answer. So a question that needs several steps works in one turn: "find events this Friday, tell me about the headliner at the top pick, and build me a warm-up set in that style" runs `find_events`, `enrich_artist`, and `build_setlist` in a single reply.

**Why three surfaces deliberately skip the agent.** Berlin Raves, Rave Set Builder, and Raves Beyond Berlin each want exactly one well-defined output. Routing any of them through a ReAct loop only adds latency, cost, and a failure mode where a small model declines to call the tool. Calling the function directly is more reliable and more honest about what each surface is.

---

## A few real workflows

1. **The first-timer's crash course.** "I have never been to a Berlin techno club, where do I start?" The agent explains the scene, names entry-level venues, describes the door, and suggests an event this weekend that suits a newcomer.
2. **The genre-matching planner.** Tell it your reference artists and ask what's on. It fetches events, weighs lineups against your profile, and ranks them closest-match first, with honest reasoning rather than hype.
3. **The era hunt.** "Something with a 90s Detroit techno sound tonight." Resident Advisor has no era filter, so the agent fetches the night, reads the lineups, finds the closest match, and tells you plainly that the era is inferred, not guaranteed.
4. **The set builder.** "Build me a deep, melodic house-party set, warmer as it goes, 16 tracks." A full arc from a 2 at the open to an 8 at the peak, each track with a reason and a preview.
5. **The label deep-dive.** "Tell me about Ostgut Ton, who records on it, and which of their artists play this weekend." KB for history, `find_events` for the weekend, `enrich_artist` for the lineups, one conversation.
6. **The Friday Telegram briefing.** Set up the GitHub Actions cron and every Friday your phone gets a short reason-first briefing that explains the standout nights, not just a list, each linked to its RA page. The app does not need to be open.

---

## Architecture

```
User prompt
    |
    v
safety.py       validate_input (length, duplicates, structural sanity)
    |            RateLimiter (configurable window per session)
    |            moderate() (classifier, score-gated)
    |            fence() (wraps untrusted content as data, not instructions)
    v
agent.py        LangGraph ReAct agent (langchain.agents.create_agent)
    |            Trusted date preamble (today, this weekend, next weekend)
    |            SqliteSaver checkpointer (cross-rerun, per-tab memory)
    |            SQLite taste profile (genres, loved artists, budget)
    |            textfmt.humanize() on the final answer
    |
    +-- explain_music()    ChromaDB RAG, allowlist filter, gap-honesty
    +-- find_events()      Resident Advisor GraphQL, clean dates and addresses
    +-- compare_events()   LLM ranking with a structured reasoning guide
    +-- enrich_artist()    Discogs primary, MusicBrainz fallback
    +-- build_setlist()    LLM energy arc + Deezer previews + YouTube links
    +-- find_club()        Deterministic Berlin club registry lookup
    +-- web_search()       Keyless DuckDuckGo fallback for KB gaps
```

**Storage.** ChromaDB vector store with local sentence-transformers embeddings (no embedding API cost), plus SQLite for conversations, taste profile, and digests.

**Observability.** LangSmith traces every LLM call and tool invocation across all four tabs when `LANGCHAIN_TRACING_V2=true`. The `@traceable` decorator covers `llm_client.chat()` (all model calls from every tab), `build_setlist` (Rave Set Builder), `compare_events` (Your Berlin Guide), and `explain_music` (Your Berlin Guide). Raves Beyond Berlin has no LLM calls so nothing to trace there.

**Models.** Five across three providers: Claude Haiku 4.5 on OpenRouter is the default, fast and works without any extra configuration. Gemini 2.5 Flash and GPT-4o Mini also route through OpenRouter. Mistral Large reuses the key that already powers the moderation call. GPT-OSS 120B on Ollama Cloud is an open-weights option that proves the same client routes to a third provider unchanged. All five live in `config.py`.

---

## The knowledge base

The KB is 27 curated markdown files, ingested into 351 ChromaDB chunks. It is Berlin-deep and music-deep on purpose: venue knowledge, party series, door culture, harm reduction, genre theory, and label history. Retrieval quality rests on the chunker in `ingest.py`, which splits each file at its section headings, prefixes every chunk with a breadcrumb from the file title and nearest heading, and overlaps adjacent chunks. The breadcrumb keeps topic words like Berghain inside a chunk even when they only appeared in the heading, and the overlap keeps a fact retrievable across a chunk boundary. `explain_music` returns `grounded=False` below a similarity threshold rather than inventing an answer.

---

## How it was built

Berlin Rave Atlas was developed in collaboration with [Claude Code](https://claude.com/claude-code). The developer set the problem, the architecture, and the quality bar; the agent helped draft modules, argue trade-offs, and test each piece against real data before it was kept. The build ran as a sequence of stages, infrastructure first, each with one deliverable and a test block that had to pass before the work was kept.

Work was matched to a model rather than defaulting to the most capable one everywhere: a fast mid-tier model for scaffolding and plumbing where there was one correct shape, the same model with a larger reasoning budget for schema and retrieval logic that is expensive to change later, and the most capable model for the prompts, the set lists, and the agent loop, where a wrong decision degrades everything downstream. The reasoning behind every architectural choice, and the alternative each one rejects, is in [CLAUDE.md](CLAUDE.md).

---

## Quick start

```bash
git clone <your-repo-url>
cd berlin-rave-atlas

cp .env.example .env
# Required: OPENROUTER_API_KEY, DISCOGS_TOKEN, MISTRAL_API_KEY

uv sync
uv run python ingest.py        # seed the knowledge base, once or after a KB edit
uv run streamlit run app.py
uv run pytest
```

> If you see 404 errors on model calls, allow providers at
> [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy).
> The default model, Claude Haiku 4.5, works without that change.

---

## Streamlit Cloud deployment

1. Push the repo to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at `app.py`, and set the Python version to 3.12 in Advanced settings (required for the pysqlite3 fix that prevents a first-load crash).
3. Paste your keys into Settings, Secrets (see `.streamlit/secrets.toml.example`).

On first load the app seeds ChromaDB from the knowledge base markdown, which takes roughly half a minute and shows a spinner. Later loads reuse the cached resource and start instantly. SQLite and ChromaDB are ephemeral on Streamlit Cloud and reset on a container restart; for durable memory, migrate SQLite to Firebase or Postgres.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | | OpenRouter key (Claude Haiku default model) |
| `DISCOGS_TOKEN` | Yes | | Discogs personal access token (artist enrichment) |
| `MISTRAL_API_KEY` | Yes | | Mistral key (moderation and the Mistral Large model) |
| `LASTFM_API_KEY` | No | | Last.fm public key (genre guard and catalogue fallback in the set builder) |
| `LANGSMITH_API_KEY` | No | | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | `false` | Set `true` to enable LangSmith |
| `OLLAMA_API_KEY` | No | | Ollama Cloud (the GPT-OSS 120B model) |
| `DEFAULT_MODEL` | No | `anthropic/claude-haiku-4.5` | One of the three ids in `config.py` |
| `TELEGRAM_BOT_TOKEN` | No | | Telegram bot for the Friday digest |
| `TELEGRAM_CHAT_ID` | No | | Target chat for the Friday digest |
| `CHROMA_DIR` | No | `data/chroma` | ChromaDB storage path |
| `SQLITE_PATH` | No | `data/rave_atlas.db` | SQLite storage path |
| `MODERATION_THRESHOLD` | No | `0.85` | Score above which a message is blocked |
| `RATE_LIMIT_REQUESTS` | No | `20` | Max requests per window per session |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Rate-limit window in seconds |

---

## Running tests

```bash
uv run pytest              # the full suite (256 tests)
uv run pytest -v           # verbose
uv run pytest tests/test_injection.py   # the injection corpus only
```

| File | Covers |
|---|---|
| `test_safety.py` | validate_input, RateLimiter, fence, moderate |
| `test_injection.py` | Injection and jailbreak vectors, false-positive music queries, fence structure |
| `test_tools.py` | explain_music, enrich_artist, find_events with city routing and clean date and address fields |
| `test_setlist.py` | build_setlist shape, energy clamping, Deezer hit and miss, LLM-failure fallback |
| `test_telegram.py` | Digest window, the reason-first briefing model fallback chain, HTML escaping, Telegram cap |
| `test_clubs.py` | The Berlin club registry lookup |
| `test_fixes.py` | Currency display, weekend-date resolution across weekdays, web_search gap-honesty |
| `test_retrieval.py` | Real ChromaDB retrieval: fact recall, scope routing, gap-honesty, allowlist filtering |

The unit tests run offline with mocked APIs. The retrieval tests run against the real local vector store and skip if it has not been built yet.

---

## Security

- **Prompt injection.** A moderation classifier with per-category score gating, not regex matching, tested against an injection and jailbreak corpus.
- **Prompt fencing.** Untrusted input is wrapped in delimiters before reaching the model, with an explicit data-not-instructions directive.
- **Trusted date context.** Date resolution happens in code and is injected as a trusted preamble, separate from the fenced user message, so relative dates resolve without mixing trusted and untrusted channels.
- **Rate limiting.** A configurable per-session rolling window.
- **Input validation.** Length bounds, a duplicate-submission guard, whitespace normalisation.
- **Retrieval allowlists.** explain_music filters by allowlist, closed by default.
- **Web-search trust boundary.** web_search results are treated as untrusted data by both the code and the prompt, never followed as instructions, and cited when used.
- **Gap-honesty.** Hallucinating events, artists, or facts is blocked by design; every tool returns a gap signal instead.

---

## Known limitations

| Limitation | Impact | Path forward |
|---|---|---|
| Resident Advisor has no official public API | find_events uses an unofficial GraphQL endpoint that can change | A maintained third-party source or the RA affiliate programme |
| Agent depth is Berlin-only by design | Other cities are browse-only in Beyond Berlin | Add city KB files and run the enrichment pipeline |
| SQLite is local | Profiles and threads reset on a stateless cloud restart | Firebase or Postgres |
| ChromaDB seeds at runtime | Roughly a half-minute cold start on first cloud load | Pre-build the vector store or use a hosted vector DB |
| In-app scheduler cannot run on a sleeping app | The Friday digest would not fire reliably in-process | The Telegram digest runs from a GitHub Actions cron instead |
| DuckDuckGo web search has no SLA | web_search can return empty under heavy use | Swap in a keyed search provider |

---

## Where it goes next

The core app covers planning, learning, and set building. The next extension adds an engagement layer on the same backend: a scene passport that stamps every club you have visited so the agent treats you as a regular, a promoter game that scores a three-DJ lineup on genre coherence and arc, an era challenge quiz on Berlin techno history, and a doorman game grounded in the door-culture content already in the knowledge base.
