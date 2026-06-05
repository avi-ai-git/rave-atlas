# Sprint 3 — Build Plan & Architecture

## Project: "Rave Atlas" — A Berlin Electronic Music Agent

*(Working name. Alternatives: Rave Atlas, Resonance, Crate, Nachtplan. Pick before you build.)*

**One-liner:** An AI agent that plans your weekend in Berlin's electronic music scene, teaches you the music, and builds your sets — grounded in live event data and a curated knowledge base, with the agent reasoning about *why* one night beats another.

This document is three things at once:
1. The **architecture** you will build on Streamlit.
2. A **phase-by-phase build guide** structured so each phase is a clean Claude Code prompt.
3. Your **review defense** — every design choice maps to a rubric criterion and pre-empts the exact issues your Sprint 1 and Sprint 2 reviewers flagged.

---

## 0. Problem definition (the first graded criterion)

The official brief leads with *Agent Purpose* and the rubric's first block is *Problem definition*. Have this nailed in one breath:

**Problem.** Berlin has the densest electronic music scene on earth, but planning a night is fragmented and high-effort: events are scattered across Resident Advisor, Instagram, and word of mouth; you can't easily tell which party matches your taste; and newcomers don't know the genres, the labels, or which artists are worth the trip. You waste Friday afternoon with twenty tabs open and still pick wrong.

**Solution.** One agent that (a) pulls the weekend's real Berlin events and *reasons* about which fit your taste and budget, (b) teaches you the music — genres, history, Berlin's labels — from a curated knowledge base, and (c) builds you a set list with playable previews. It remembers your taste across sessions and proactively briefs you every Friday.

**Why an agent (not a chatbot or a static app).** The task needs runtime decisions — fetch events, or enrich an artist, or retrieve theory, or build a set — chosen per request. That branching is exactly what an agent does and a static chain can't.

**Target users.** (1) You and people like you — locals who go out and want better-matched nights with less effort. (2) Newcomers and tourists who love electronic music but don't know Berlin's scene. (3) Curious learners who want the music education without committing to a course.

---

## 1. Why this design targets 90+ (read this first)

Sprint 3 is two modules fused: **AI Agents** (intro → chains → tools/APIs/automation → long-term memory → current tech → building with agents) and **Agentic Coding** (build it *with* Cursor/Claude Code, and be able to explain every line). The single most heavily weighted rubric criterion in both your past sprints was **demonstrating understanding in the 1-1 review** — not the artifact. So this plan optimises for two things together: a technically complete agent, and a codebase small and clean enough that you can defend every decision.

### The reviewer fix-list (baked into the architecture from day one)

Your two reviewers handed you the Sprint 3 checklist. Every item below was a deduction last time. None of them will exist this time because the architecture includes them as first-class components.

**From Sprint 1 (Interview App, 99):**
- No rate limiting → **add a rate limiter** (session timestamp + request count, reject over threshold). OWASP LLM10.
- Regex-only injection denylist → **use a real moderation classifier** (Mistral moderation API) gated on category scores, not substring matches.
- No tests → **pytest parametrized tests** on every pure function and tool.
- One giant `app.py` → **modular structure** (prompts / safety / client / tools / UI separated).
- No logging → **structured logging to stdout** (model, status, latency, tool calls, injection hits).
- Lower-bound-only pins → **pyproject.toml + uv + committed lockfile**.

**From Sprint 2 (RAG Coach, 93):**
- Commit rarely → **commit often, push full git history** (shows the agentic build process — also an Agentic Coding module signal).
- Knowledge base hard-coded in repo → **extract KB into a data store** (SQLite or Firebase) that also persists long-term user sessions.
- No metrics/observability → **Langfuse or structlog** for latency, token usage, model distribution.
- Hard-coded models/URLs → **everything in env vars / config**.
- Top-to-bottom `app.py` → **`main()` entry point + small testable functions**.
- Query-translation prompt had no examples → **few-shot examples in prompts**.
- Retrieval used blacklists → **per-tool allowlists** for retrieval scope.
- Manual injection testing → **automated injection test suite** (OWASP corpus + jailbreak attempts + false-positive checks).
- No caching → **in-memory cache** on repeated queries / API calls.
- `create_react_agent` imported from wrong place → **import from `langchain.agents`**, not `langgraph.prebuilt`.
- Incomplete type hints → **full type hints**.

If you build the architecture below, you have already answered all 20 of these before a reviewer opens the repo.

### What carries forward that reviewers *praised* (keep doing it)
- The **ReAct agent pattern** with runtime tool selection — your Sprint 2 reviewer called it "the right abstraction." Reuse it.
- **Semantically distinct, well-sized tools** with specific docstrings that steer the agent without extra routing.
- **Honest known-limitations** documentation with migration paths.
- **Grounding output in verified evidence** (gap-honesty). The agent must say "no match this weekend" rather than invent a party.

---

## 2. The product: three features + one automation

One Streamlit app, three views, one background job.

### Feature A — This Weekend in Berlin (the Concierge)
You describe a vibe and constraints ("hypnotic techno, Friday, under €20, near Kreuzberg"). The agent fetches live Resident Advisor events, enriches each with artist/label history from Discogs, and recommends nights — **reasoning about why one fits you better than another** (genre match, artist lineage, room size, price, your saved taste profile).

### Feature B — How Electronic Music Works (the Explainer)
A RAG-grounded music-education view. Curated knowledge base covering: genre families (techno, house, psytrance, dubstep, trance, ambient, breakbeat), BPM/feel signatures, a short history of Berlin's scene and key labels (Tresor, Ostgut Ton/Berghain, BPitch), and how a track is structured. This is your domain knowledge base — the rubric explicitly rewards a "relevant knowledge base for your domain."

### Feature C — Build Me a Set (the Crate)
Give it a vibe or seed artists; the agent proposes a tracklist with an energy arc and a one-line reason per track, then attaches a **free playable link** to each: Deezer 30-second preview (embeddable, no painful auth) plus a YouTube search URL (zero-key). Output is a shareable set list.

### Automation — The Weekend Digest
A scheduled job (Friday morning) compiles the weekend's Berlin parties into a briefing: the parties, their genres, a little history on the headlining artists, and the agent's reasoning on standouts. This satisfies the **Automation** lecture *and* your explicit ask. It writes the digest to the data store so the app can show it, and can optionally email it to you.

---

## 3. System architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Streamlit UI (app.py)           │
                    │   Tab: Weekend  │  Tab: Learn  │  Tab: Crate │
                    │            thin views, main()                │
                    └───────────────────────┬─────────────────────┘
                                            │
                        ┌───────────────────▼───────────────────┐
                        │   Safety gate (safety.py)              │
                        │  validation → moderation → rate limit  │
                        └───────────────────┬───────────────────┘
                                            │
                        ┌───────────────────▼───────────────────┐
                        │   LangGraph ReAct Agent (agent.py)     │
                        │   reason → pick tool → observe → loop  │
                        │   + SqliteSaver long-term memory       │
                        └───────┬───────┬───────┬───────┬────────┘
                                │       │       │       │
              ┌─────────────────▼─┐ ┌───▼────┐ ┌▼──────┐ ┌▼────────────┐
              │ find_events       │ │enrich_ │ │explain│ │ build_setlist│
              │ (RA GraphQL)      │ │artist  │ │_music │ │ (LLM+Deezer/ │
              │                   │ │(Discogs│ │(RAG / │ │  YouTube)    │
              │ compare_events    │ │ MBz)   │ │Chroma)│ │              │
              └─────────┬─────────┘ └───┬────┘ └───┬───┘ └──────┬───────┘
                        │               │          │           │
                   ┌────▼───────────────▼──────────▼───────────▼────┐
                   │  llm_client.py (OpenRouter) · cache · structlog │
                   └────────────────────┬───────────────────────────┘
                                        │
                   ┌────────────────────▼───────────────────────────┐
                   │  Data store (SQLite): KB embeddings (Chroma),   │
                   │  user taste profile, session memory, digests    │
                   └─────────────────────────────────────────────────┘

   Background:  weekend_digest.py  ── scheduled Fri AM ──►  writes digest to store
```

**The agent loop in one sentence:** the LLM reads the user's request and its memory, decides which tool to call, the harness runs it, feeds the result back, and the loop repeats until it can answer — the ReAct pattern your Sprint 2 reviewer praised, now over live external APIs.

---

## 4. Tech stack

| Layer | Choice | Why / carry-forward |
|---|---|---|
| UI | Streamlit | Required; you know it; deployable free |
| Agent framework | LangGraph + LangChain ReAct agent | Reviewer-praised pattern; reuse Sprint 2 |
| Agent import | `from langchain.agents import create_react_agent` | Fixes Sprint 2 deduction |
| LLM gateway | OpenRouter (you have the key) | Carry-forward; model in env var |
| Embeddings | local sentence-transformers | No API cost; carry-forward |
| Vector store | ChromaDB | Carry-forward |
| Memory | LangGraph SqliteSaver | Fixes Sprint 2 "resets on reload"; satisfies Memory lecture |
| Data store | SQLite (taste profile, digests, sessions) | Fixes Sprint 2 "extract KB to data store" |
| Moderation | Mistral moderation API | Fixes Sprint 1 regex deduction |
| Logging | structlog → stdout | Fixes both sprints |
| Observability | Langfuse (optional, bonus) | Fixes Sprint 2 metrics gap |
| Packaging | pyproject.toml + uv + lockfile | Fixes Sprint 1 deduction |
| Tests | pytest (parametrized) | Fixes both sprints |
| Scheduling | APScheduler or a cron-style script | Automation lecture + your ask |

---

## 5. Data sources & APIs (all free)

| Tool | Source | Auth | Limit | Returns |
|---|---|---|---|---|
| `find_events` | Resident Advisor GraphQL | none (unofficial) | be polite, cache | Berlin events: name, venue, date, lineup, genre, price, link |
| `enrich_artist` | Discogs API | free token | 60 req/min | artist releases, labels, genre/style lineage |
| `enrich_artist` (fallback) | MusicBrainz | none | 1 req/sec | artist, label, area metadata |
| `explain_music` | your curated KB | none | — | grounded genre/history/theory answers |
| `build_setlist` links | Deezer API | none for previews | fair use | 30-sec preview MP3 URL per track |
| `build_setlist` links | YouTube search URL | none | unlimited | `youtube.com/results?search_query=...` |

**Notes that will come up in review:** Spotify's audio-features endpoint was deprecated for new apps in Nov 2024 — that's *why* you're not using it; mention this, it shows current-tech awareness. RA has no official public API, so you call its GraphQL endpoint directly and **cache aggressively + degrade gracefully** if it changes — document this as a known limitation with a migration path (a maintained scraper or Apify), exactly the honest-limitations style your reviewers rewarded.

---

## 6. Repository structure (modular from line one)

```
rave-atlas/
├── pyproject.toml          # deps + metadata (uv)
├── uv.lock                 # committed lockfile
├── .env.example            # OPENROUTER_API_KEY, DISCOGS_TOKEN, MISTRAL_API_KEY, model names
├── README.md               # problem, architecture, run, limitations+migration paths
├── config.py               # loads env, model names, base URLs — NO hard-coded values
├── app.py                  # thin Streamlit UI, main(), tab routing only
├── agent.py                # LangGraph ReAct assembly, memory wiring
├── llm_client.py           # OpenRouter client + error mapping + cache
├── safety.py               # input validation, moderation, rate limiting, prompt fencing
├── logging_config.py       # structlog setup
├── memory.py               # SqliteSaver + taste-profile read/write
├── prompts/
│   ├── system.py           # agent system prompt (data-as-data fencing)
│   ├── setlist.py          # few-shot examples
│   └── compare.py          # reasoning prompt for event comparison
├── tools/
│   ├── events.py           # find_events, compare_events (RA)
│   ├── artists.py          # enrich_artist (Discogs/MusicBrainz)
│   ├── music_kb.py         # explain_music (Chroma retrieval, per-tool allowlist)
│   └── setlist.py          # build_setlist (+ Deezer/YouTube links)
├── automation/
│   └── weekend_digest.py   # scheduled Fri job → writes digest to store
├── knowledge_base/         # curated markdown (genres, history, labels, theory)
│   ├── genres_techno.md
│   ├── genres_house.md
│   ├── genres_psytrance.md
│   ├── genres_dubstep.md
│   ├── berlin_scene_history.md
│   ├── labels.md
│   └── track_anatomy.md
├── data/                   # gitignored: chroma db, sqlite, sessions
└── tests/
    ├── test_safety.py      # validation, rate limit, moderation
    ├── test_tools.py       # parametrized tool behaviour (mocked APIs)
    ├── test_injection.py   # OWASP corpus + jailbreaks + false positives
    └── test_setlist.py     # link building, energy-arc shape
```

**Gitignore:** `data/`, `.env`, any `*_private/`. Mirror the Sprint 2 discipline.

---

## 7. Knowledge base design (the graded "domain KB")

Write these yourself — your taste *is* the moat and reviewers reward a genuine domain KB. Each file: what the genre/topic is, BPM and rhythmic signature, emotional feel, 3-5 defining artists/labels, how to recognise it, and (for history) the Berlin-specific story. Keep entries chunk-friendly (200-400 words). This corpus is what `explain_music` retrieves over, and it's also what the agent uses to reason about genre fit in `compare_events`. Honest scope: cover the genres you actually know well; the KB being real and opinionated beats being broad and thin.

---

## 8. Security, observability, testing (the deduction-killers)

**Safety gate** runs before any model or API call, in order: (1) input validation (empty / too short / too long / duplicate), (2) Mistral moderation classifier gated on category scores, (3) rate limiter (last-submit timestamp + rolling request count in session state, reject over threshold). All three are the explicit Sprint 1 fixes.

**Prompt fencing:** wrap all user content and all tool/API output in clearly labelled fences and instruct the model to treat it as data, never as instructions — the pattern your Sprint 1 reviewer praised, now extended to tool outputs (which is where agents get injected).

**Logging:** structlog to stdout on every call — model, tool name, status code, latency, token usage, cache hit/miss, moderation hits. **Observability (bonus):** pipe to Langfuse for a metrics dashboard.

**Tests:** parametrized pytest on safety helpers, each tool (with mocked API responses so tests are deterministic and free), the setlist link builder, and a dedicated injection suite running an OWASP prompt-injection corpus + known jailbreaks (DAN etc.) + legitimate-query false-positive checks. Target the pure functions — they're where the edge-case bugs live.

**Caching:** in-memory cache keyed on query/tool-args so repeated questions and repeated API calls in a session don't re-spend credit or re-hit RA/Discogs.

---

## 9. Long-term memory + automation (your bonus points)

**Memory:** `SqliteSaver` persists conversation across page reloads (the exact Sprint 2 migration path the reviewer named). On top of that, a **taste profile** in SQLite: your genres, loved/blocked artists, budget ceiling, preferred areas, thumbs-up history. The agent reads it at the start of every session and updates it when you react to recommendations — so "find me something Friday" gets personalised without you re-stating preferences. This is the headline bonus and most learners skip it.

**Automation:** `weekend_digest.py` runs Friday morning (APScheduler, or a script you trigger on a schedule). It calls `find_events` for Fri–Tue, enriches headliners via `enrich_artist`, asks the agent to rank and explain standouts, and writes the digest to the store for the app to display (optional: email it to you). This is a concrete, demoable answer to the Automation lecture and your weekend-planning use case.

---

## 10. Build phases (each phase = one Claude Code session)

Build in order. Run each phase's tests before moving on. Commit after every phase (full git history is a graded signal this time). Suggested models: **planning/architecture → Opus, "think hard"**; **implementation → Sonnet**; **debugging → Sonnet or Opus with thinking**.

**Phase 0 — Scaffold.** pyproject.toml + uv, repo structure, config.py, logging_config.py, .env.example, gitignore, empty test files. *Deliverable:* `uv run streamlit run app.py` shows three empty tabs. *Prompt seed:* "Scaffold a Python project named after-dark using uv and pyproject.toml with this module structure [paste §6]. Set up structlog in logging_config.py and a config.py that loads all settings from env vars with no hard-coded models or URLs."

**Phase 1 — Music KB + explain_music (RAG core).** Write the knowledge_base markdown, build the Chroma index with local sentence-transformers, implement `explain_music` with a per-tool allowlist. *Deliverable:* the Learn tab answers genre/history questions, grounded. *This phase alone is a complete, defensible RAG app — your safety net.*

**Phase 2 — The agent + safety gate.** Assemble the LangGraph ReAct agent (`from langchain.agents import create_react_agent`), wire `explain_music` as its first tool, add the safety.py gate and prompt fencing. *Deliverable:* you chat with an agent that decides when to retrieve.

**Phase 3 — Live events: find_events + compare_events.** RA GraphQL fetch with caching and graceful degradation; `compare_events` reasoning prompt. *Deliverable:* the Weekend tab returns real Berlin events and explains why one fits.

**Phase 4 — enrich_artist.** Discogs (token) + MusicBrainz fallback; feed lineage into event reasoning. *Deliverable:* recommendations cite artist/label history.

**Phase 5 — build_setlist + Crate tab.** LLM tracklist with energy arc + per-track reason; attach Deezer preview + YouTube link. *Deliverable:* shareable set list with playable previews.

**Phase 6 — Memory + taste profile.** SqliteSaver + SQLite taste profile read/update; personalise recommendations. *Deliverable:* preferences persist across reloads. *(Hard optional task.)*

**Phase 7 — Weekend digest automation.** Scheduled Fri job → digest to store → shown in app. *Deliverable:* a weekend briefing generated without you asking. *(Medium optional task.)*

**Phase 8 — Tests + observability + polish.** Fill the test suite (incl. injection corpus), add caching, optional Langfuse, write the README with honest limitations + migration paths, deploy to Streamlit Cloud. *Deliverable:* green tests, deployed URL.

---

## 11. Optional tasks mapped to the OFFICIAL list (need 2 medium + 1 hard)

These map to the brief's exact numbered optional tasks, so the bonus claim is airtight. You build well past the minimum — call out the cleanest three in your review.

**Hard (you hit several — name one as primary):**
- **#1 Agentic RAG** — `explain_music` retrieval invoked by the agent at runtime *is* agentic RAG. **(primary hard)**
- **#5 Integrate external data sources** — RA events, Discogs, MusicBrainz, Deezer enrich the agent's knowledge live.
- **#2 LLM observability** — Langfuse/LangSmith dashboard (Phase 8).
- **#7 Deploy to cloud** — Streamlit Cloud deployment.
- **#4 Learns from feedback** — taste profile updates from your thumbs-up/down to improve future recs.

**Medium (you hit several — name two):**
- **#3 Long-term/short-term memory** — SqliteSaver + taste profile (Phase 6). **(primary medium)**
- **#4 One more external-API tool** — you ship four+ external tools. **(primary medium)**
- **#7 Feedback loop with ratings** — rate recommendations, stored to profile.
- **#6 Caching** — in-memory cache (Phase 8).
- **#9 Multi-model support** — OpenRouter already routes Anthropic and Ollama runs through Gemma and ChatGPT opensource models; expose a model picker (also easy-task #3).
- **#1 Token usage & cost display** — show per-query tokens/cost in the UI.
- **#2 Retry logic** — exponential-backoff retries on tool/LLM calls.

**Easy (cheap polish, grab a few):** #1 ask Claude to critique (do this and quote it in your README); #2 agent personality; #3 model picker; #4 temperature/top-p sliders; #5 in-app help.

That's comfortably 1 hard + 2 medium *with a deep reserve* — top of the bonus criterion. Don't build all of them; pick the three you can defend best and list the rest as "implemented" extras.

---

## 12. Review defense — questions you must be able to answer

Have a crisp answer for each (the 1-1 is the heaviest-weighted criterion):
- Why an **agent (ReAct)** and not a static chain or plain RAG? (Conditional logic: it decides whether to fetch events, enrich artists, or retrieve theory at runtime.)
- How does **function calling / tool selection** work in your agent, end to end?
- How does your **memory** work, and what's the difference between session memory and the taste profile?
- What are the **security risks** (injection via tool output, unbounded consumption) and exactly how you mitigate each?
- Where could it **fail** (RA endpoint changes, empty weekend, API rate limits) and how it degrades honestly?
- What would you **improve** next (data store → Firebase, eval harness, more genres)?
- Why **not** Spotify audio features? (Deprecated Nov 2024 — shows current-tech awareness.)
- **When do you use prompt engineering vs RAG vs an agent?** (Explicit rubric criterion.) Prompt engineering shapes a single response; RAG grounds answers in a knowledge base when facts must be sourced (your `explain_music`); an agent is for multi-step tasks needing runtime tool choice (your whole app). Your project uses all three, deliberately layered — say exactly that.
- **Do you actually use "the OpenAI API"?** Yes — OpenRouter is OpenAI-API-compatible (same SDK/params), and it also gives you free multi-model support (OpenAI, Anthropic, Gemini) behind one key. Mention this; it satisfies the OpenAI-API requirement *and* an optional task at once.
- What are the **agent types** and how does yours differ? (ReAct vs plan-and-execute vs multi-agent — know the trade-offs; you chose ReAct for runtime branching.)

If you can answer these line by line, you're in "Excellent — meets or exceeds" territory on the explanation criteria.

---

## 13. Risks & cut-lines (don't over-scope — the module warns against this)

If time runs short, ship in this priority: Phases 1-3 are the core (a working agent over a real KB and live events — already a strong pass). Phase 5 (Crate) is the crowd-pleaser. Phases 6-7 are the bonus. Phase 8 is non-negotiable polish (tests + deploy + README). A smaller agent you fully understand and can defend beats a sprawling one you can't — that single principle is what separates a 90 from a 99.
