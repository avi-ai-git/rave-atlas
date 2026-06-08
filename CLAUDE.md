# CLAUDE.md, Berlin Rave Atlas

This document serves two readers. It onboards a future Claude Code session before it touches any file, and it records, for anyone reading the repository, how the project was designed, what was chosen, and why each alternative was rejected. It is written to be defended out loud, not skimmed.

Berlin Rave Atlas was built by a developer pairing with Claude Code as a coding collaborator: the developer set the problem, the architecture, and the quality bar, and used the agent to draft modules, argue trade-offs, and test each piece against real data before keeping it. The decisions below are the developer's; the agent was the tool that made iterating on them fast.

---

## 1. What it is and the problem it solves

Berlin has the densest electronic music scene on earth, and planning a night in it is fragmented, high-effort work. Events are scattered across Resident Advisor, Instagram, and word of mouth. You cannot tell which party matches your taste without already knowing the scene, and newcomers do not know the genres, the labels, or which artists are worth the trip.

The app is one agent that does four things: it fetches live Berlin events and reasons about which match your taste and budget rather than just listing them; it teaches the music, genres, history, and Berlin's labels from a curated knowledge base; it builds a set with a deliberate energy arc and playable previews; and it remembers your preferences across sessions and can send a Friday briefing.

It is for three people: Berlin locals who want better-matched nights with less research, newcomers who love the music but do not know the scene, and curious listeners who just want to understand it.

---

## 2. How it was built, in order

The build ran as a sequence of stages, each with one deliverable and a test block that had to pass before the work was kept. The order was deliberate: infrastructure first, so that content and features could be tested against real behaviour as they were added, not after.

1. **Configuration and provider client.** `config.py` loads every setting from the environment so nothing is hard-coded, and `llm_client.py` wraps three OpenAI-compatible providers behind one `chat()` call with caching, retry, and cost estimation. Built first because every other module depends on a working, observable model call.
2. **Safety boundary.** `safety.py` (validation, rate limiting, a moderation classifier, input fencing) and `logging_config.py` (structured JSON logs). Built before the agent so that no later code path could reach the model without passing the gate.
3. **Retrieval pipeline.** `ingest.py` chunks the knowledge base into ChromaDB, and `tools/music_kb.py` queries it with gap-honesty. Built before the knowledge content so the content could be tested against real retrieval as it was written.
4. **The tools.** The seven domain tools in `tools/`, each independently importable with its own test block.
5. **The agent.** `agent.py` composes the safety gate, the provider client, the prompts, the tools, and persistent memory into one `run_agent()` entry point. Built once its dependencies were each tested in isolation, so a failure here is a composition bug, not a unit bug.
6. **Memory.** `memory.py` adds the LangGraph checkpointer and the SQLite taste profile.
7. **The interface.** `app.py`, a thin Streamlit UI that routes to the agent or to a direct tool call per tab and renders the result.
8. **The knowledge base, in two passes.** A first pass wrote seed content to validate the chunker and the metadata schema; a second pass deepened it with first-hand Berlin venue knowledge, door culture, harm reduction, genre theory, and label history.
9. **Automation.** The in-app digest, the standalone Telegram digest on a GitHub Actions cron, and the KB enrichment pipeline.

The knowledge base content came last on purpose. Domain depth is the one thing the tooling cannot scaffold, so the retrieval infrastructure was built first to test each piece of content against real queries as it landed.

### Models used while building

Work was matched to a model rather than defaulting to the most capable one for everything. Scaffolding, API wrappers, and plumbing used a fast mid-tier model because there was one correct shape and no design decision to make. Schema design, retrieval logic, and UI state used the same model with a larger reasoning budget because those are expensive to change once downstream code depends on them. User-visible output, the prompts and the set lists, and the agent loop in `agent.py` used the most capable model, because a wrong decision there degrades everything a person actually reads or that every other component depends on. The git history records which work used which model.

### Lessons carried in from earlier iterations

This is a refined build, and the most useful input was the weaknesses earlier versions shipped with. Each lesson became an engineering decision here rather than a patch.

| Lesson | What this version does | Where |
|---|---|---|
| A regex denylist for prompt injection is trivially bypassed | A moderation classifier with score-based gating across harm categories | `safety.py` |
| No rate limiting leaves the door open to abuse | A configurable rolling-window limiter per session | `safety.py` |
| Retrieval blacklists miss anything unanticipated | Per-tool allowlists, closed by default | `tools/music_kb.py` |
| A single monolithic UI module is untestable | Small modules, each independently importable with a test block | every module |
| No tests and no logging make regressions invisible | A real pytest suite and structured JSON logging on every call | `tests/`, `logging_config.py` |
| Lower-bound-only dependency pins drift and break | Compatible-range pins and a committed lockfile | `pyproject.toml`, `uv.lock` |
| No observability means no insight into cost or latency | LangSmith tracing on every LLM call and key tool call across all four tabs | `llm_client.py`, `agent.py`, `tools/` |
| Hard-coded models and URLs are brittle | Everything loads from the environment | `config.py` |

---

## 3. Architecture decisions, and the alternative each one rejects

**A ReAct agent, not a static chain or plain retrieval.** The app has three task shapes (event lookup, music education, set building) plus cross-cutting enrichment. A static chain fixes the tool path when the code is written; a retrieval-only app can only answer from documents. A ReAct agent reads the message and decides at runtime which tools to call, in what order, and when it has enough to answer. That runtime branching is the whole reason a chain or a plain RAG app cannot solve this well.

**Three surfaces deliberately bypass the agent.** Berlin Raves, the Rave Set Builder, and Raves Beyond Berlin each want exactly one well-defined output: a digest plus a raw RA browse, a single fully enriched set, or one city's listings. Routing any of them through a ReAct loop adds latency, adds cost, and adds a real failure mode where a small model decides not to call the tool. Calling the function directly is the honest, reliable choice. Picking the right altitude per surface, agent or retrieval or direct call, is the architecture, not a default.

**LangGraph SqliteSaver for memory.** Streamlit re-runs the whole script on every interaction, so in-memory state resets constantly. The SqliteSaver checkpointer persists conversation threads to disk keyed by session and per tab, so a reload resumes exactly where it left off. The taste profile is a separate concern, a user model that improves with feedback, so it lives in its own SQLite table. Two tables, one file, no extra infrastructure, which is the right weight for a single-user app.

**ChromaDB with local embeddings.** The knowledge base is embedded with a local sentence-transformers model (`all-MiniLM-L6-v2`). A hosted embedding API would add per-call cost, a network dependency, and a rate limit for no benefit at this corpus size. Local embedding is free, offline, and fast enough, and the vectors persist on disk. The ingestion pipeline is idempotent, so re-running it after a KB edit is safe.

**Heading-aware, overlapped chunking.** Retrieval quality lives or dies on chunking. `ingest.py` splits each markdown file at its section headings, prefixes every chunk with a breadcrumb drawn from the file title and the nearest heading, and overlaps adjacent chunks within a section. The breadcrumb puts the topic words inside the chunk, so a chunk about a door policy still carries the word Berghain even when that word only appeared in the heading. The overlap keeps a fact retrievable when it falls across a chunk boundary. A naive fixed-size splitter loses both, which is why generic chunking returns the wrong section for a query like "Berghain door policy".

**Three providers behind one client.** `llm_client.py` routes to OpenRouter, Mistral, or Ollama Cloud by the model's provider, all through the OpenAI SDK shape. One call site, no provider branching scattered through the code, plus an in-memory cache, exponential-backoff retry, and cost estimation.

**A moderation classifier, not regex, for injection defence.** A denylist is bypassed by obfuscation, paraphrasing, or zero-width characters. A moderation API scores harm categories regardless of surface form, and the gate is score-based. On top of that, user input is fenced as data and the system prompt is told never to follow instructions inside fenced content. Defence in depth, not a single string match.

**Gap-honesty everywhere.** `explain_music` returns `grounded=False` below a similarity threshold, `find_events` returns an empty list with a logged status when Resident Advisor fails, and `web_search` returns `grounded=False` on any failure. The agent surfaces these signals instead of inventing an answer, because a fabricated party or a made-up label is worse than an honest "I do not have that".

**House typography enforced in code.** Models ignore the instruction to avoid em dashes often enough that `textfmt.humanize()` normalises every piece of user-facing model output (chat answers, set-list reasons, event rankings, the Telegram briefing) to straight quotes and plain punctuation. The rule holds regardless of which model wrote the text.

**Clean event data from a messy source.** Resident Advisor returns full ISO timestamps and inconsistent address strings. `tools/events.py` parses the timestamp into a readable date and time, normalises the address into one line, and prefers the curated registry address for known Berlin venues. The UI never shows a raw timestamp.

---

## 4. When the project uses prompt engineering, retrieval, or an agent

All three, layered on purpose.

| Technique | Where | Why |
|---|---|---|
| Prompt engineering | `prompts/` | Shapes every generation: the persona, the few-shot energy-arc examples, the ranking guide. Author-time, single step. |
| Retrieval (RAG) | `tools/music_kb.py` | Grounds music answers in the curated KB. Used when facts must be sourced, not invented. |
| Agent (ReAct) | `agent.py`, Your Berlin Guide tab | Multi-step reasoning with runtime tool selection. |
| Direct tool call | Berlin Raves, Rave Set Builder, and Raves Beyond Berlin tabs | Single, well-defined work where a deterministic call beats asking a model to decide to make it. |

---

## 5. Runtime models

The default runtime model is **Claude Haiku 4.5** on OpenRouter: the fastest and cheapest tier, and the one model that works on a default OpenRouter account without privacy-policy configuration, so a fresh clone runs out of the box. The sidebar exposes four alternatives: **Gemini 2.5 Flash** and **GPT-4o Mini** also route through OpenRouter, **Mistral Large** reuses the same key that powers the moderation call, and **GPT-OSS 120B** on Ollama Cloud is an open-weights model that proves the same client routes to a third provider unchanged. All five are defined in `config.py` and route through the same `llm_client.chat()` call without any provider-specific branching in the calling code.

---

## 6. Module map

```
berlin-rave-atlas/
  app.py                  Thin Streamlit UI, four tabs (Your Berlin Guide, Berlin Raves, Rave Set Builder, Raves Beyond Berlin) and routing only
  agent.py                LangGraph ReAct agent assembly + run_agent()
  config.py               All settings from environment, no hard-coded values
  llm_client.py           OpenRouter + Mistral + Ollama client, cache, retry, cost
  logging_config.py       structlog JSON setup
  memory.py               SqliteSaver checkpointer + SQLite taste profile + digests
  safety.py               validate_input, moderate, RateLimiter, fence
  textfmt.py              humanize(), house typography on all model output
  ingest.py               Heading-aware, overlapped KB chunking into ChromaDB
  prompts/
    system.py             Agent persona, date resolution, tool-routing rules
    setlist.py            Set-list prompts: Pass 1 arc schema (role, bpm_target, energy), Pass 2 track selection + set_story
    compare.py            Event-ranking reasoning guide
  tools/
    music_kb.py           explain_music, RAG with allowlist and gap-honesty
    events.py             find_events (RA GraphQL, city-aware) + compare_events
    artists.py            enrich_artist, Discogs primary, MusicBrainz fallback
    setlist.py            build_setlist, two-pass arc + Last.fm genre guard + BPM/role/set_story output, Deezer + YouTube
    clubs.py              find_club, deterministic Berlin club registry lookup
    club_registry.py      The Berlin club table (addresses, official links)
    web.py                web_search, keyless DuckDuckGo fallback, gap-honest
  automation/
    weekend_digest.py     In-app APScheduler digest (local and demo use)
    weekend_telegram.py   Standalone reason-first Telegram digest (GitHub Actions cron)
    kb_enrich.py          Reddit and web KB enrichment pipeline
    club_scraper.py       Optional official-site event scraper
  .github/workflows/
    weekend-digest.yml    Cron that runs the Telegram digest, app-independent
  knowledge_base/         27 curated markdown files, ingested to ChromaDB (351 chunks)
  data/                   Gitignored, chroma/ and the SQLite database
  tests/                  pytest suite, 256 tests, offline by default
```

---

## 7. The seven tools

The agent picks among these at runtime.

1. `explain_music` retrieves grounded context from the knowledge base, with an optional allowlist of doc types.
2. `find_events` fetches live events from Resident Advisor and returns clean, normalised event data.
3. `compare_events` ranks fetched events against the taste profile with plain-language reasoning, not numeric scores.
4. `enrich_artist` pulls labels, genres, and releases from Discogs, falling back to MusicBrainz.
5. `build_setlist` generates an energy-arc set list in two passes. Pass 1 has the model plan the arc, choose artists, assign a function role (opener, build, peak, sustain, resolution, closer), and estimate a genre-appropriate BPM target per position. Between passes, a Last.fm genre guard rejects any artist whose top tag is non-electronic and who carries no electronic tags in their top five, so only genuine electronic artists reach the catalogue call. Deezer is then queried with an `artist:"X"` precise search; for artists absent from Deezer, Last.fm `artist.getTopTracks` provides a catalogue fallback. Pass 2 has the model select a specific track from that verified list at temperature 0.2 (selection from a fixed menu, not generation) and write a two-to-three sentence set story. This structure eliminates hallucinated track titles structurally. Each track in the output carries: title, artist, role, energy level, BPM (from Deezer if available, otherwise the Pass 1 estimate), a one-line reason, a 30-second Deezer preview, and a YouTube link.
6. `find_club` looks up a Berlin venue's official site, events page, and address from a curated registry, a deterministic fact lookup rather than a vector search.
7. `web_search` is the keyless fallback for current facts outside the knowledge base, with results treated as untrusted data.

---

## 8. The knowledge base

The `knowledge_base/` markdown is Berlin-deep and music-deep on purpose. An earlier iteration carried thin one-paragraph primers for many European cities; they were removed because a 150-word file on a city scores worse than a live web search and dilutes the Berlin retrieval that is the point of the app. What remains is genuinely opinionated and specific: venue knowledge, party series, door culture, harm reduction, genre theory, and label history. A deep, narrow KB beats a broad, thin one for this use case. The pipeline is idempotent, so re-running `uv run python ingest.py` after any edit is safe.

---

## 9. Conventions for a future session

- Every setting comes from `config.py`. Do not hard-code a model id, URL, price, threshold, or API key anywhere else. This includes `LASTFM_API_KEY`: the default is baked into `config.py` as a fallback, but the live value must always be read from the environment.
- Every user-facing string that a model produced must pass through `textfmt.humanize()`. No em dashes or en dashes anywhere, in code, in the UI, or in the knowledge base.
- Every module keeps its `if __name__ == "__main__"` test block runnable in isolation.
- Tools return a gap signal rather than inventing data. Preserve that.
- After any knowledge base edit, re-run `uv run python ingest.py`, then `uv run pytest`.

---

## 10. Known limitations and where they go next

| Limitation | Impact | Path forward |
|---|---|---|
| Resident Advisor has no official public API | `find_events` uses an unofficial GraphQL endpoint that can change | A maintained third-party source or the RA affiliate programme |
| Agent depth is Berlin-only by design | Other cities are browse-only in Beyond Berlin | Add city KB files and run the enrichment pipeline |
| SQLite is local | Profiles and threads reset on a stateless cloud restart | Firebase or Postgres |
| ChromaDB seeds at runtime | A cold start of roughly half a minute on the first cloud load | Pre-build the vector store or use a hosted vector DB |
| The in-app scheduler cannot run on a sleeping app | The Friday digest would not fire reliably in-process | The Telegram digest runs from a GitHub Actions cron instead |
| DuckDuckGo web search has no SLA | `web_search` can return empty under heavy use | Swap in a keyed search provider |

---

## 11. What is next, an engagement layer

The core app is a utility. The next extension adds features that bring people back on a daily or weekly cadence, sharing the same backend: a scene passport that stamps every club you have visited so the agent treats you as a regular; a promoter game that scores a three-DJ lineup on genre coherence, scene credibility, and arc; an era challenge, a fast quiz on Berlin techno history; and a doorman game that teaches how a real door reads a clubber, grounded in the door-culture content already in the KB. These work because the knowledge base has real depth; a shallow KB makes a game that breaks in two rounds, which is why the content work came first.
