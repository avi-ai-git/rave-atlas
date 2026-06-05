"""
Rave Atlas — event comparison reasoning prompt.

Used by tools/events.py (Phase 7) to rank a list of events against the user's
taste profile. The output is a ranked list with plain-language reasoning per
event — not numeric scores. A reviewer reading the output should understand
*why* the agent thinks one night fits and another does not.
"""

from __future__ import annotations

# Placeholders: {events_json}, {taste_profile_json}
COMPARE_PROMPT: str = """\
You are ranking Berlin electronic music events against a specific user's \
taste profile. Your job is to recommend the best fits and honestly flag \
mismatches — not to oversell every event.

## The taste profile (read this first)

This describes what the user likes and what they want to avoid. If a field \
is empty, ignore that criterion — do not penalise an event for missing data.

```json
{taste_profile_json}
```

## The events to rank

```json
{events_json}
```

## Ranking criteria — weight roughly in this order

1. **Genre match** — does the event's primary genre fit `preferred_genres`? \
Penalise if it's in `blocked_genres`.
2. **Artist / label lineage** — do the headliners come from labels or scenes \
the user has loved before (`loved_artists`, or genre families they prefer)? A \
Klockworks artist for a user who loves Ben Klock is a strong signal.
3. **Budget** — is the price under `budget_ceiling`? Going over is a hard \
mark-down, not a soft preference.
4. **Area** — is the venue in `preferred_areas`? Friedrichshain is a 30-min \
S-bahn from Kreuzberg; mention it as a tradeoff if relevant.
5. **Novelty / discovery** — if everything else is equal, slightly favour an \
event that introduces an artist or label the user has not encountered, \
especially if it's adjacent to their preferred genres. Do not push novelty \
that conflicts with the profile.

## How to write each reasoning block

Three short sentences max:
- **Sentence 1:** the core fit reason (genre + lineup match) OR core mismatch.
- **Sentence 2:** the supporting detail (price, area, label lineage).
- **Sentence 3 (optional):** the honest tradeoff, if any.

Do not assign numeric scores. The reasoning IS the recommendation. Be \
specific: name the artist, the label, the venue. "Klockworks-aligned" beats \
"techno-leaning". "€18 under your €25 ceiling" beats "affordable".

## Output schema (must be valid JSON)

{{
  "ranked_events": [
    {{
      "rank":         <integer, starting at 1>,
      "event_name":   "<the event name from input, verbatim>",
      "fit_summary":  "<one short line summarising the verdict>",
      "reasoning":    "<2-3 sentences as described above>",
      "tradeoff":     "<one line of honest downside, or null if there is none>"
    }},
    ...
  ]
}}

Rank every input event (do not drop any). If an event is a clear bad fit, \
rank it last with an honest "doesn't fit your profile because…" reasoning. \
Honesty about mismatches earns the user's trust — silently dropping events \
does not.

Return ONLY the JSON object. No preamble, no markdown fences.
"""


def build_compare_prompt(events: list[dict], taste_profile: dict) -> str:
    """
    Return the COMPARE_PROMPT filled with the events list and taste profile.

    Both are serialised as JSON so the LLM has structured input rather than
    prose that could be misinterpreted.
    """
    import json
    return COMPARE_PROMPT.format(
        events_json=json.dumps(events, indent=2, ensure_ascii=False),
        taste_profile_json=json.dumps(taste_profile, indent=2, ensure_ascii=False),
    )
