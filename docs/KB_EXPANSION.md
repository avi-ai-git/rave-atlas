# Knowledge Base Expansion Plan

The Rave Atlas knowledge base lives in `knowledge_base/` as markdown. Its value
comes from being **personal, opinionated, and specific**, not broad and thin. A
machine can scaffold structure; only you can add the first-hand Berlin knowledge
that makes it worth using. This doc is the plan for growing it.

## Two ways to add content

### 1. Write it yourself (preferred for canon)

Drop a new `.md` file into `knowledge_base/` and run:

```bash
uv run python ingest.py
```

Ingestion is idempotent (upsert), so re-running after any edit is instant and safe.
Files in the root are treated as curated canon. Use your own research tools to
draft comprehensive, sourced notes, then edit them into your voice before saving.

### 2. Auto-enrich from Reddit and the web (drafts to review)

```bash
uv run python automation/kb_enrich.py --dry-run   # preview summaries, write nothing
uv run python automation/kb_enrich.py             # write + re-ingest
```

This fetches configured sources (Reddit public JSON, no key needed; optional web
pages), runs each through a strict LLM cleaning pass, and writes attributed files
to `knowledge_base/community/`. These are tagged `doc_type="community"` so
retrieval can tell crowd-sourced notes from your canon. **Always review what lands
there before trusting it.** Configure sources at the top of `automation/kb_enrich.py`.

## Topic checklist (gaps worth filling)

Current canon: techno, house, psytrance, dubstep, Berlin scene history, labels,
track anatomy, rave culture, rave wellness, electronic music theory.

Suggested additions, in rough priority order:

- [ ] **How to rave (first-timer guide)** expand beyond the culture file: ticket
      buying, RA vs door, group size, what to do if turned away, solo raving.
- [ ] **What to wear** Berlin black, layering, KitKat dress codes, practical vs
      fashion, seasonal (open-air vs winter club), shoe advice.
- [ ] **City scene profiles** Amsterdam, London, Paris, Barcelona, Belgrade,
      Tbilisi, and how each connects to Berlin's lineage. One file per city or one
      comparative file.
- [ ] **Genre deep-dives still missing** trance (uplifting/psy/hard), drum & bass,
      electro, EBM/industrial, hard techno/hardgroove, ambient/downtempo, gabber.
- [ ] **Electronic music history timeline** Kraftwerk to Detroit to Chicago to
      Berlin, the 1990s rave explosion, the Love Parade, Klubsterben.
- [ ] **Production basics** synthesis, drum machines (909/808/303), sampling,
      arrangement, the breakdown/drop, sidechaining.
- [ ] **The business** labels, vinyl vs digital, Bandcamp, residencies, booking,
      how DJs actually get paid.
- [ ] **Festivals** Awakenings, Dekmantel, Time Warp, Sonar, Atonal, open-airs.

## Sourcing guidance

- **Prefer primary and reputable secondary sources**: RA features, label sites,
  artist interviews, books (e.g. *Energy Flash*, *Lost and Sound*), Wikipedia for
  dates/facts. Use community sources (Reddit) for lived experience and etiquette,
  not for hard facts.
- **Summarise, never paste.** Keep the KB in your own words to stay clear of
  copyright and to keep one consistent voice.
- **Be specific.** "Tresor runs a harder, more industrial sound than Watergate's
  melodic deep house" beats "there are many great clubs."
- **Cite era and place.** Scene facts age; note when something was true.
- **No em or en dashes** anywhere in the KB, to match the app's house style.

## How retrieval uses it

`explain_music` embeds each chunk with a local sentence-transformers model and
retrieves by cosine similarity. It returns `grounded=False` when nothing is close
enough, and the agent then either says so or (now) falls back to `web_search`.
More high-quality, specific chunks means more questions land `grounded=True`.
