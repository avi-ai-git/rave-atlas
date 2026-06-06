# Rave Atlas

> Berlin's electronic music agent. Weekend event concierge, music education, set-list builder, and Europe-wide rave browse in one app.

[![Tests](https://img.shields.io/badge/tests-162%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![Built with Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-orange)](https://claude.ai/code)

---

## Who is this for?

**Berlin regulars** who go out every weekend but spend Friday afternoon with 20 RA tabs open, still unsure which party matches their sound.

**First-timers and tourists** who love electronic music but don't know Berghain from about blank, or techno from deep house, or what a Klockworks night actually sounds like.

**Curious listeners** who want to understand the music — genre history, BPM signatures, how a DJ set is structured — without sitting through a Wikipedia spiral.

---

## What does it do?

Four things, each with its own tab:

**Raves in Berlin** is an AI agent that talks to you. Ask it "what's on this Friday under 15 euros near Kreuzberg" or "I want something hypnotic and 90s-influenced tonight" and it fetches live Resident Advisor events, reasons over them against your taste profile, and tells you which nights fit and which don't, with a direct link to every event. It learns from the ratings you give it.

**Rave Wiki** is a knowledge base for the scene. Ask it anything about electronic music genres, Berlin's club history, record labels (Ostgut Ton, Klockworks, Tresor Records, Innervisions), DJ technique, or how a techno track is structured. When the knowledge base doesn't have something, it searches the web and tells you which answer came from where.

**Set Builder** builds you a tracklist with a deliberate energy arc. Give it a vibe, a time of night, a number of tracks, or a venue name, and it returns artist and title per track with a one-line reason for each position, plus 30-second Deezer previews and YouTube links. Useful for a party playlist, a comedown set, or just understanding how a DJ shapes a night.

**Rave Parties in Europe** is a direct browse of Resident Advisor events for any European city by date, genre, price, and area. No AI — just the listings, honestly. Berlin is in there, but so is Amsterdam, Paris, Barcelona, Warsaw, Istanbul, and 60+ other cities.

---

## Full potential — what it can do

The agent tab is a ReAct agent: it picks tools at runtime based on what you ask, chains them, and synthesises a response. This means questions that require multiple steps work in one turn:

- "Find me events this Friday, tell me about the headliner at the top pick, and build me a warm-up set in that style"  
  → find_events, enrich_artist, build_setlist — three tools, one reply
- "What label does Marcel Dettmann record on, and is he playing Berlin this weekend?"  
  → enrich_artist, find_events — grounded in real data
- "I like what I heard at Tresor last month, what's the label's sound and who should I watch next?"  
  → explain_music (KB), web_search (recent artist news)

---

## Innovative use cases and workflows

1. **The first-timer's crash course.** Ask "I've never been to a Berlin techno club, where do I start?" The agent explains the scene, names entry-level venues (Watergate, Ritter Butzke, OHM), describes what to expect at the door, and suggests an event this weekend that fits a newcomer.

2. **The genre-matching night planner.** Tell the agent your reference artists ("I love Four Tet, Floating Points, Actress") and ask what's on this weekend. It fetches events, enriches lineups against your taste profile, and ranks them from closest match to furthest — with honest reasoning, not hype.

3. **The 90s or era-specific night hunt.** Ask for "something with a 90s Detroit techno sound tonight." RA has no era filter, so the agent fetches events, reasons over lineups and genre tags to find the closest match (a classic techno night, a vinyl-only party, a retro-influenced DJ), and tells you honestly that the era match is inferred, not guaranteed.

4. **The budget-maximised weekend.** "What's free or under 10 euros this weekend, and is any of it actually worth going to?" The agent fetches, filters, and ranks against your profile — no need to open RA manually.

5. **The before-you-go artist prep.** "Tell me about Surgeon before I go see him tonight." The agent pulls label history, genre lineage, and notable releases from Discogs and MusicBrainz, giving you the context to understand what you're about to hear.

6. **The party set builder.** "Build me a 3-hour midnight set for a house party, deep and melodic, gets warmer as it goes, 20 tracks." Returns a full arc from 2/10 energy at opening to 8/10 at peak, with artist, title, reason per track, and playable previews.

7. **The comedown set.** "It's 7am Sunday, I just got home from Berghain, build me something ambient that winds down over an hour." Six tracks, energy arc 5 to 1, Burial-adjacent textures. Previews included.

8. **The genre learning ladder.** "Explain the difference between minimal techno, industrial techno, and EBM, with examples." The Rave Wiki retrieves from the curated knowledge base and links you to tracks on YouTube so you can hear the distinctions rather than just read them.

9. **The label deep-dive.** "Tell me about Ostgut Ton, who records on it, and what events feature their artists this weekend." KB retrieval for label history, find_events for the weekend, enrich_artist to check lineups — all in one conversation.

10. **The multi-city Europe trip.** Use the Rave Parties in Europe tab to browse Amsterdam, Berlin, and Prague in one session. Each city shows live RA listings for your dates. Plan three nights across three cities without opening three browser tabs.

11. **The Friday Telegram digest.** Set up the GitHub Actions cron and every Friday morning you get a Telegram message with the Berlin weekend lineup, each event linked directly to its RA page. The app doesn't need to be open.

12. **The knowledge base for your own city.** Edit `automation/kb_enrich.py` with subreddits and web sources for your local scene, run it, and the app's Rave Wiki now knows your city's clubs, labels, and history. The ingestion pipeline is idempotent — re-run it any time.

13. **The crate-digger's reference.** "I found a record from roughly 1994, dark industrial kick, 138 BPM, one-bar loop. What label era does this sound like?" The Rave Wiki explains the Tresor/Underground Resistance sound, the Roland 909 signature, and the Detroit-to-Berlin pipeline that defined that production aesthetic.

14. **The touring DJ homework.** "I'm playing a warm-up set at Fabric next month, it's a deep house night. Prep me." The agent uses web_search for recent Fabric lineups, the KB for deep house theory, and build_setlist for a reference arc — all cited and sourced.

15. **The regular's weekly ritual.** Rate events after you go ("Like" or "Not for me"). The agent updates your taste profile in SQLite and the next Friday's recommendations are noticeably more accurate. After a month of use, the ranking is personal rather than generic.

16. **The producer's structural reference.** "What BPM range does a Tresor-style techno track sit at, and how is it structured in terms of intro, build, drop, and outro?" The KB has dedicated track anatomy content — useful for producers learning club-ready arrangement.

17. **The venue comparison before booking.** "Compare Watergate and about blank for a melodic techno night — what's the vibe difference, the sound system, the door policy?" The KB holds venue-specific commentary gathered from real experience, not press releases.

18. **The scene history explainer.** "How did Berlin become the centre of techno after reunification?" Full Rave Wiki answer covering the Tresor founding, the Berghain lineage, the squat-culture roots, and the specific geography (Mitte warehouses, Friedrichshain power stations) that shaped the sound.

---

## Architecture

```
User prompt
    |
    v
safety.py       validate_input (length, duplicates, structural sanity)
    |            RateLimiter (20 req / 60 s per session)
    |            moderate() (Mistral classifier, score-gated, OWASP LLM01)
    |            fence() (wraps untrusted content as data, not instructions)
    v
agent.py        LangGraph ReAct agent  (langchain.agents.create_agent)
    |            Date preamble injected as trusted context (today, this weekend, next weekend)
    |            SqliteSaver checkpointer (cross-rerun conversation memory)
    |            SQLite taste profile (genre preferences, loved artists, budget)
    |
    +-- explain_music()    ChromaDB RAG, allowlist filter, gap-honesty
    +-- find_events()      Resident Advisor GraphQL, client-side filters, city-aware
    +-- compare_events()   LLM with structured reasoning rubric
    +-- enrich_artist()    Discogs primary, MusicBrainz fallback
    +-- build_setlist()    LLM energy arc + Deezer previews + YouTube URLs
    +-- web_search()       DuckDuckGo keyless fallback for KB gaps
```

**Storage:** ChromaDB vector store (local sentence-transformers, no API cost) + SQLite (conversations + taste profile + digests)

**Observability:** LangSmith traces every LLM call and tool invocation when `LANGCHAIN_TRACING_V2=true`

**Models:** Claude Haiku 4.5 (default, fast, works out of the box) + Mistral Large (alternative, same key used for moderation)

---

## How it was built — 17-phase agentic coding methodology

Built phase-by-phase using [Claude Code](https://claude.ai/code). Each phase had one deliverable, a model matched to the work, a thinking budget matched to the decision complexity, and a test block that had to pass before the phase was committed.

| Model allocation | Phases | Rationale |
|-----------------|--------|-----------|
| Claude Sonnet 4.6 | 0-4, 6-7, 9, 11-14, 17 | Scaffolding, API wrappers, CRUD, fixes, feature additions |
| Claude Opus 4.7 | 5 (prompts), 8 (setlist), 10 (agent orchestration) | User-visible quality and orchestration correctness |

Sonnet built roughly 80% of the codebase. Opus was reserved for the three phases where reasoning quality determines output quality. The git log records which model built each component.

### Build sequence

| Phase | What shipped |
|-------|-------------|
| 0 | Scaffold: `pyproject.toml`, `config.py`, `logging_config.py`, blank `app.py`, `CLAUDE.md` |
| 1 | Knowledge base markdown files + `ingest.py` (ChromaDB ingestion pipeline) |
| 2 | `tools/music_kb.py`: RAG retrieval with allowlist filter and gap-honesty |
| 3 | `safety.py`: input validation, Mistral moderation (score-gated), rate limiter, prompt fencing |
| 4 | `llm_client.py`: OpenRouter + Ollama Cloud client, in-memory cache, exponential-backoff retry, cost estimation |
| 5 | `prompts/system.py`, `prompts/setlist.py`, `prompts/compare.py`: agent persona, few-shot set-list examples, event-ranking reasoning rubric |
| 6 | `tools/artists.py`: `enrich_artist` via Discogs primary and MusicBrainz fallback |
| 7 | `tools/events.py`: `find_events` (RA GraphQL, city-aware) + `compare_events` (LLM-ranked) |
| 8 | `tools/setlist.py`: `build_setlist` with energy arc, Deezer 30-second previews, YouTube links |
| 9 | `memory.py`: LangGraph `SqliteSaver` + SQLite taste profile + feedback loop |
| 10 | `agent.py`: ReAct agent assembly, safety gate, checkpointer, six tools |
| 11 | `automation/weekend_digest.py` + `automation/weekend_telegram.py` + GitHub Actions cron |
| 12 | `app.py` (full): four tabs, model picker, tone radio, temperature and top-p sliders, event cards, ratings, cost display |
| 13 | `tests/`: pytest suite (safety, tools, injection corpus, setlist, Telegram) |
| 14 | `README.md`, LangSmith observability wiring, Streamlit Cloud deploy prep |
| 17 | Phase 17 batch: pysqlite3 Streamlit Cloud crash fix, date resolution rewrite, currency display, Mistral chat model, keyless web search (`tools/web.py`), KB enrichment pipeline (`automation/kb_enrich.py`), full UI copy pass (de-emoji, no dashes, humanised text), region filter regroup, digest moved with RA links, 162 tests |

### Why each architectural choice was made

**ReAct agent over static chain:** the app has three fundamentally different task types (event lookup, music education, set building) plus cross-cutting enrichment. A static chain hardcodes the path at author-time; the agent decides at runtime which tools to call, in what order, and when it has enough to answer.

**Direct tool calls for Set Builder and Explore:** these tabs always want exactly one thing (one set, one city's listings). Routing through the agent adds latency, cost, and a failure mode where the model might decide not to call the tool. Calling directly is more reliable and honest about what the feature is.

**LangGraph `SqliteSaver`:** Streamlit reruns the script on every interaction, resetting in-memory state. `SqliteSaver` persists conversation threads to disk keyed by `session_id` so the page can reload and the agent resumes exactly where it left off. The taste profile is a separate concern (it's a user model, not conversation history) stored in a second SQLite table.

**ChromaDB with local sentence-transformers:** embedding the KB with an API adds cost, a network dependency, and a rate limit. The local `all-MiniLM-L6-v2` model is free, runs offline, and is fast enough for the 48-file knowledge base (148 chunks). Vectors persist in `data/chroma/` across restarts; the pipeline is idempotent so re-running `ingest.py` after any KB edit is safe.

**Mistral moderation (score-gated) over regex:** a regex denylist is trivially bypassed by obfuscation or paraphrasing. A classifier assigns probabilities across harm categories regardless of surface form. Score-based gating is the OWASP LLM01 mitigation.

**Gap-honesty:** `explain_music` returns `grounded=False` when similarity is below threshold. `find_events` returns an empty list with a logged status on RA failure. `web_search` returns `grounded=False` on any failure. The agent surfaces these signals rather than inventing answers. Hallucinated events or fabricated music facts are worse than "I don't have data for that."

---

## Quick start

```bash
# 1. Clone
git clone <your-repo-url>
cd rave-atlas

# 2. Copy and fill secrets
cp .env.example .env
# Required: OPENROUTER_API_KEY, DISCOGS_TOKEN, MISTRAL_API_KEY

# 3. Install (uv manages the virtualenv automatically)
uv sync

# 4. Seed the knowledge base (once, or after editing knowledge_base/*.md)
uv run python ingest.py

# 5. Run
uv run streamlit run app.py

# 6. Tests (all offline, no API calls needed)
uv run pytest
```

> **OpenRouter note:** if you see 404 errors on model calls, visit
> [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) and enable providers.
> The default model (Claude Haiku 4.5) works without this change.

---

## Streamlit Cloud deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py`, set **Python version to 3.12** in Advanced settings (required for the pysqlite3 fix that prevents a crash on first load)
3. In **Settings → Secrets**, paste your keys (see table below)

On first load the app automatically seeds ChromaDB from the knowledge base markdown files. This takes roughly 30 seconds and shows a spinner. Subsequent loads reuse the cached resource and start instantly.

SQLite and ChromaDB are ephemeral on Streamlit Cloud (reset on container restart). For durable memory across sessions, migrate SQLite to Firebase or PostgreSQL.

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | | OpenRouter API key (Claude Haiku default model) |
| `DISCOGS_TOKEN` | Yes | | Discogs personal access token (artist enrichment) |
| `MISTRAL_API_KEY` | Yes | | Mistral API key (moderation + Mistral Large model) |
| `LANGSMITH_API_KEY` | No | | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | `false` | Set to `true` to enable LangSmith |
| `OLLAMA_API_KEY` | No | | Ollama Cloud (open-source models, commented out by default) |
| `DEFAULT_MODEL` | No | `anthropic/claude-haiku-4.5` | Override the default model |
| `TELEGRAM_BOT_TOKEN` | No | | Telegram bot for Friday digest |
| `TELEGRAM_CHAT_ID` | No | | Target chat for Friday digest |
| `CHROMA_DIR` | No | `data/chroma` | ChromaDB storage path |
| `SQLITE_PATH` | No | `data/rave_atlas.db` | SQLite storage path |
| `MODERATION_THRESHOLD` | No | `0.7` | Mistral score above which to block a message |
| `RATE_LIMIT_REQUESTS` | No | `20` | Max requests per window per session |
| `RATE_LIMIT_WINDOW_SECONDS` | No | `60` | Rate-limit window in seconds |

---

## Running tests

```bash
uv run pytest              # all 162 tests
uv run pytest -v           # verbose
uv run pytest tests/test_injection.py   # OWASP injection corpus only
```

| File | Tests | What is covered |
|------|-------|----------------|
| `test_safety.py` | 28 | `validate_input`, `RateLimiter`, `fence`, `moderate` |
| `test_injection.py` | 57 | OWASP/jailbreak vectors, false-positive music queries, fence structural tests |
| `test_tools.py` | 31 | `explain_music` (ChromaDB mock), `enrich_artist` (Discogs/MusicBrainz mock), `find_events` + city routing (RA mock) |
| `test_setlist.py` | 15 | `build_setlist` shape, energy clamping, Deezer hit/miss, LLM failure fallback |
| `test_telegram.py` | 11 | Weekend digest HTML, escaping, Telegram-cap truncation, send paths |
| `test_fixes.py` | 20 | Currency display (EUR and non-EUR), `weekend_dates` correctness on all weekdays including Friday/Saturday regression, web_search gap-honesty, Mistral model wiring |

All tests run offline. Every external API is mocked.

---

## Module map

```
rave-atlas/
├── app.py                   Thin Streamlit UI: tab routing and rendering only
├── agent.py                 LangGraph ReAct agent + run_agent()
├── config.py                All settings from env vars; no hard-coded values
├── llm_client.py            OpenRouter + Mistral + Ollama Cloud; cache, retry, cost
├── logging_config.py        structlog JSON to stdout
├── memory.py                SqliteSaver checkpointer + SQLite taste profile
├── safety.py                validate_input, moderate (Mistral), RateLimiter, fence
├── ingest.py                Chunks KB markdown -> ChromaDB (run once or after KB edit)
├── prompts/
│   ├── system.py            Agent persona + date resolution + tool routing instructions
│   ├── setlist.py           SETLIST_PROMPT with few-shot energy-arc examples
│   └── compare.py           Event-ranking reasoning rubric
├── tools/
│   ├── music_kb.py          explain_music: RAG, allowlist, gap-honesty
│   ├── events.py            find_events (RA GraphQL, city-aware) + compare_events (LLM)
│   ├── artists.py           enrich_artist: Discogs -> MusicBrainz fallback
│   ├── setlist.py           build_setlist: LLM arc + Deezer + YouTube
│   └── web.py               web_search: keyless DuckDuckGo fallback, gap-honest
├── automation/
│   ├── weekend_digest.py    APScheduler Fri 09:00 -> Fri-Tue briefing -> SQLite (in-app)
│   ├── weekend_telegram.py  Standalone Berlin digest -> Telegram (run by GitHub Actions)
│   └── kb_enrich.py         Reddit + web KB ingestion: fetch, LLM-clean, write to community/
├── .github/workflows/
│   └── weekend-digest.yml   Cron (Fri 07:00 UTC) -> weekend_telegram, app-independent
├── knowledge_base/           48 curated markdown files, ingested to ChromaDB
│   ├── genres_techno.md      + genres_house, genres_psytrance, genres_dubstep
│   ├── berlin_scene_history.md
│   ├── berlin_berghain_ecosystem.md  + berlin_history_deep_cuts, berlin_labels_current
│   ├── berlin_club_venues.md + berlin_club_doors, berlin_party_series, berlin_smaller_clubs
│   ├── labels.md, track_anatomy.md, music_theory_electronic.md
│   ├── electronic_music_genres_bpm.md, rave_culture_history.md
│   ├── club_etiquette_logistics.md, harm_reduction.md
│   ├── city_primer_amsterdam.md  + 28 other European city primers
│   └── community/           Auto-enriched drafts from kb_enrich.py (gitignored)
├── docs/
│   └── KB_EXPANSION.md      Sourcing checklist for expanding the knowledge base
└── tests/
    ├── conftest.py
    ├── test_safety.py
    ├── test_injection.py
    ├── test_tools.py
    ├── test_setlist.py
    ├── test_telegram.py
    └── test_fixes.py
```

---

## Security

- **OWASP LLM01 (Prompt Injection):** Mistral moderation API with per-category score gating, not regex matching. Tested against 19 injection vectors including OWASP corpus and common jailbreaks.
- **Prompt fencing:** all untrusted input is wrapped in `=== BEGIN DATA ===` delimiters before reaching the LLM, with an explicit data-not-instructions directive.
- **Trusted date context:** date resolution happens in code and is injected as a trusted preamble, separate from the fenced user message, so relative dates work reliably without mixing untrusted and trusted channels.
- **Rate limiting:** 20 requests per 60 seconds per session, configurable.
- **Input validation:** length bounds, duplicate-submission guard, whitespace normalisation.
- **Retrieval allowlists:** `explain_music` uses allowlists (not blacklists) for doc_type filtering — closed by default.
- **Web search trust boundary:** results from `web_search` are treated as untrusted external data by both the code and the agent's prompt. The agent is instructed never to follow instructions in web results and to cite the source when using them.
- **Gap-honesty throughout:** hallucinating events, artists, or facts is explicitly blocked; every tool returns a gap signal rather than inventing data.

---

## What's next — capstone extension

The core app covers event planning, music education, and set building. The next extension adds an engagement layer: features that bring people back on a daily or weekly cadence rather than just when planning a night out. All four share the same backend as the current app (same KB, same agent, same SQLite profile), surfaced in a new "Play" tab.

**Scene Passport (Phase 20)** — a digital stamp book of every club you have been to. The more you check in, the more the agent treats you as an insider rather than a newcomer. Long-term: a world map with a pin for every club across every city you have ever visited. Ravers travel specifically to go to clubs and there is no good app for recording that journey. The data model built in Phase 20 is already correct for it; the map is a rendering layer added on top.

**The Promoter Game (Phase 21)** — you book 3 real DJs for a real Berlin venue (example brief: "Tresor, Saturday midnight, 300 capacity, 15 euro door, hypnotic techno") and the agent scores your lineup on musical logic, scene credibility, and arc coherence. Replayable with a different venue and genre each round. Losing is educational because the agent explains its reasoning in scene terms.

**Era Challenge (Phase 21)** — the agent picks a year from Berlin techno history and you name 3 artists or tracks from that year. Pub quiz format, two minutes, streak mechanic. Short enough to do daily; teaches real scene history with every round.

**The Doorman (Phase 22)** — you play the Berghain Türsteher. The agent describes a fictional clubber in three sentences. You decide in or out. The agent tells you whether a real doorman would agree and why. The most shareable feature in the roadmap; requires a comprehensive Berghain door-culture knowledge base before it can be built credibly.

---

## Known limitations

| Limitation | Impact | Migration path |
|------------|--------|---------------|
| RA has no official public API | `find_events` uses an unofficial GraphQL endpoint that may change | Maintained third-party scraper or RA affiliate programme when available |
| Agent core is Berlin-only | Deep knowledge (venues, labels, history) is Berlin-grade; other cities in Explore tab are browse-only | Add city-specific KB files and run kb_enrich.py for other scenes |
| SQLite is local-only | Taste profiles and conversation history reset on Streamlit Cloud container restart | Migrate to Firebase Realtime DB or PostgreSQL |
| ChromaDB seeded at runtime | Roughly 30-second cold start on Streamlit Cloud | Pre-commit a built ChromaDB or switch to a hosted vector DB |
| In-app APScheduler can't reliably run on Streamlit Cloud | The app sleeps when idle | Telegram digest is decoupled to a GitHub Actions cron that runs whether or not the app is awake |
| Spotify audio-features API deprecated (Nov 2024) | No programmatic BPM or energy data | Deezer public API (used) and AcousticBrainz as alternatives |
| DuckDuckGo web search has no rate-limit guarantee | `web_search` may return empty results under heavy use | Replace with a Serper or SerpAPI key if consistent web results are needed |

---

## Prompt engineering vs RAG vs agent — where each is used

| Technique | Where | Why |
|-----------|-------|-----|
| Prompt engineering | `prompts/system.py`, `prompts/setlist.py`, `prompts/compare.py` | Shapes every LLM generation: persona, few-shot arc examples, event-ranking rubric. Applied at author-time to a single step. |
| RAG | `tools/music_kb.py` via `explain_music` | Grounds music-theory answers in the curated KB. Used when facts must be sourced, not invented. `grounded=False` is surfaced to the user. |
| Agent (ReAct) | `agent.py` via `run_agent()`, Berlin tab, Rave Wiki tab | Used where the task requires multi-step reasoning with runtime tool selection. The agent decides at runtime which of six tools to call based on what the user asked. |
| Direct tool call | Set Builder tab, Rave Parties in Europe tab | Used where the task is always exactly one thing: one set, one city's listings. More reliable and cheaper than asking a model to decide to make a deterministic call. |

The Crate and Go-International tabs deliberately do not use the agent. Calling `build_setlist` or `find_events` directly eliminates the latency, the token cost, and the failure mode where a small model declines to call the tool. Picking the right altitude per feature — agent, RAG, prompt, or plain function — is the architecture.

---

*Built with [Claude Code](https://claude.ai/code)*
