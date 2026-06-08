"""
Rave Atlas, set-list generation prompts.

Two-pass architecture:
  Pass 1 (ARTISTS_PROMPT / build_artists_prompt):
    LLM plans the arc and selects artists. No track titles yet — the model
    does what it is good at: genre knowledge, scene positioning, arc structure.

  Between passes (tools/setlist.py):
    For each selected artist, Deezer is queried for their real, streamable
    catalogue. This list is injected into pass 2 so the model selects from
    verified options rather than recalling from memory.

  Pass 2 (TRACKS_PROMPT / build_tracks_prompt):
    LLM assigns a specific track to each arc position, choosing only from
    the Deezer catalogue supplied. Temperature is very low (0.2) because
    selection from a fixed menu, not generation.

The old SETLIST_PROMPT is kept for backward compatibility but is no longer
called by build_setlist; it is used only by the smoke test.

Few-shot examples cover four seed types so the model has a pattern for
each common arc shape: hypnotic techno, deep melodic warm-up, industrial
peak, and closing ambient comedown.
"""

from __future__ import annotations

# ── Pass 1: artist and arc planning ──────────────────────────────────────────
#
# Placeholders: {seed}, {n}

ARTISTS_PROMPT: str = """\
You are a Berlin DJ planning the arc for a set. Select the artists and their \
roles. Do NOT choose track titles yet — the titles will be assigned in a \
second step using real, verified Deezer tracks.

## Output schema (must be valid JSON)

{{
  "title": "<short evocative set title>",
  "positions": [
    {{
      "position": <integer starting at 1>,
      "artist": "<exact artist name as it appears on their releases>",
      "role": "<opener|build|peak|sustain|resolution|closer>",
      "energy": <integer 1-10>,
      "bpm_target": <integer: approximate BPM target for this position. Typical ranges: deep house 118-124, tech house 124-130, techno 130-142, industrial techno 138-148, psytrance 140-150>,
      "reason": "<one line: why this artist belongs at this position>"
    }},
    ...
  ]
}}

Rules:
- Plan all {n} positions. Every position must have a role and an energy value.

- **Genre coherence is absolute.** Every artist must belong to the seed genre \
or a directly adjacent sub-genre. No genre drift at any position. If the seed \
is hypnotic techno, every artist must be a techno or recognised techno-variant \
act, not progressive house, not trance, not drum and bass.

- **Smooth energy arc.** No two adjacent positions may differ by more than 2 \
energy units. Plan the arc shape first (e.g. 3, 4, 5, 6, 8, 8, 7, 6), then \
assign artists to fit. An arc that reads 3, 7, 4, 9, 2 is incoherent and wrong.

- **Real, documented artists.** Prefer artists with verified releases on \
Klockworks, Ostgut Ton, Tresor, Innervisions, Rush Hour, Internasjonal, SUED, \
Livity Sound, Hemlock, or equivalent labels for the seed genre.

- **Reason must be scene-specific.** Name a label affiliation or scene \
position ("Klockworks regular", "Ostgut Ton resident", "Innervisions core act", \
"Berghain resident"). Generic reasons like "great artist" or "good fit" are \
not allowed.

Return ONLY the JSON object. No explanation, no markdown fences, no preamble.

---

Seed: {seed}
Number of positions: {n}
"""


# ── Pass 2: track selection from verified Deezer catalogue ───────────────────
#
# Placeholders: {title}, {arc_block}
# arc_block is built by build_tracks_prompt() from the pass-1 output +
# Deezer catalogue per artist.

TRACKS_PROMPT: str = """\
You are completing a set list. The arc plan below shows each position with \
its artist and — where available — the artist's real tracks confirmed on \
Deezer streaming.

**Critical rule:** For each position that has an "Available on Deezer" list, \
you MUST pick a title from that list. Do not use a title that is not on the \
list. Do not invent a title. If the list has only one option, use it.

For positions with no Deezer list (artist not found), pick the artist's most \
well-known commercial release. If you are not 100% certain the title exists, \
choose a different track by that artist that you are certain about.

## Arc plan and available tracks

{arc_block}

## Output schema (must be valid JSON)

{{
  "title": "{title}",
  "set_story": "<2-3 sentences describing the arc: what world this set inhabits, how the energy moves across the set, and what the listener feels at the end versus the start>",
  "tracks": [
    {{
      "artist": "<artist name, exactly as in the arc plan>",
      "title": "<track title — MUST come from the Available on Deezer list if one is given>",
      "reason": "<one line: why THIS specific track at THIS position — texture, BPM character, arc role>",
      "energy": <integer, match the arc plan unless you have a specific musical reason to adjust by 1>
    }},
    ...
  ]
}}

Additional rules:
- Reasons must describe the track, not just the artist. "Stripped percussive \
loop, holds the floor locked at peak" is a reason. "Great track" is not.
- Do not use em dashes or en dashes. Use commas or full stops.
- Keep the arc smooth. Adjacent tracks should not differ by more than 2 energy \
units.

Return ONLY the JSON object. No explanation, no markdown fences, no preamble.
"""


def build_artists_prompt(seed: str, n: int) -> str:
    """Return the ARTISTS_PROMPT filled with the user's seed and track count."""
    return ARTISTS_PROMPT.format(seed=seed, n=n)


def build_tracks_prompt(title: str, positions: list[dict]) -> str:
    """
    Build the TRACKS_PROMPT from the pass-1 arc plan enriched with Deezer
    catalogues. Each position becomes a labelled block like:

        Position 1 | Role: opener | Energy: 4/10
        Artist: Marcel Dettmann
        Available on Deezer: Plain, Gradient, Float, Seduction, Silhouette
        Arc reason: Berghain resident, stripped kick, opens space without committing energy.

    When the artist has no Deezer catalogue, the Available line reads
    "(not found — pick a real, certain title from your knowledge)".
    """
    lines: list[str] = []
    for pos in positions:
        p = pos.get("position", "?")
        artist = pos.get("artist", "")
        role = pos.get("role", "")
        energy = pos.get("energy", "?")
        arc_reason = pos.get("reason", "")
        available: list[str] = pos.get("available_tracks") or []

        lines.append(f"Position {p} | Role: {role} | Energy: {energy}/10")
        lines.append(f"Artist: {artist}")
        genre_warning = pos.get("genre_warning", False)
        if genre_warning:
            lines.append(
                "Available on Deezer: (GENRE WARNING: Last.fm does not classify "
                "this artist as electronic music. Replace this artist with a "
                "verified Berlin electronic music act in the same role and genre. "
                "Do not use this artist.)"
            )
        elif available:
            lines.append(f"Available on Deezer: {', '.join(available)}")
        else:
            lines.append(
                "Available on Deezer: (not found on Deezer or Last.fm — only use "
                "this artist if you are 100% certain a specific track title exists "
                "on streaming platforms. If uncertain, replace with a more "
                "mainstream artist in the same genre whose tracks you are certain about.)"
            )
        if arc_reason:
            lines.append(f"Arc reason: {arc_reason}")
        lines.append("")

    arc_block = "\n".join(lines).strip()
    return TRACKS_PROMPT.format(title=title, arc_block=arc_block)


# ── Legacy single-pass prompt (kept for smoke test / fallback) ────────────────
#
# Placeholders: {seed}, {n}

SETLIST_PROMPT: str = """\
You are a Berlin DJ and crate-digger building a set list for the user.

A good set has a deliberate **energy arc**: warm-up tracks open space and \
invite the floor in, peak tracks lock the room, comedown tracks let everyone \
breathe. Energy is 1 (ambient, almost silence) to 10 (peak-time saturation). \
A set without an arc is just a playlist.

## Output schema (must be valid JSON)

{{
  "title": "<short evocative title for the set>",
  "tracks": [
    {{
      "artist": "<artist name>",
      "title": "<track title>",
      "reason": "<one line: why this track in this position>",
      "energy": <integer 1-10>
    }},
    ...
  ]
}}

Rules:
- Exactly the number of tracks the user asks for. For a 16-track set, assume \
roughly 4 minutes per track, about 1 hour. Plan the arc accordingly: a full \
opening phase (tracks 1-4), a build phase (5-8), a peak phase (9-12), and \
a resolution and close phase (13-16).

- **Genre coherence is absolute.** Every single track must belong to the seed \
genre or a directly adjacent sub-genre only. If the seed is "hypnotic techno", \
every track must be techno or a recognised techno variant (industrial techno, \
dark techno, Detroit techno). Do not include progressive house, trance, drum \
and bass, or any other genre, ever. If the seed is "deep house", every track \
must be deep house, soulful house, minimal deep, or organic house, not tech \
house, not melodic techno. If you cannot find a real track in the correct \
genre for a given position, pick a different artist in that genre rather than \
crossing into another genre. Genre drift is the most damaging mistake in set \
building and will make the set unusable.

- **Energy arc must be smooth.** Plan the arc shape first, before choosing \
tracks. No two adjacent tracks should differ by more than 2 energy units. \
An arc of 3, 7, 4, 9, 2, 8 is incoherent and wrong. A correct arc reads \
like a deliberate curve: 3, 4, 5, 7, 8, 7, 6 (warm-up to peak to resolution).

- **Artist selection reasoning is required.** For each track the reason field \
must say: (a) why this artist fits the genre, naming their label affiliation \
or scene position ("Klockworks regular", "Innervisions core act", "Berghain \
resident"), (b) what the energy level means in context ("first peak, the room \
locks in"), and (c) what makes this arc position right for them. Generic \
reasons are not allowed. "Great track" is not a reason. "Stripped \
Detroit-influenced kick, right for a 5/10 build moment before the first peak" \
is a reason.

- **Choose real, documented artists with releases on relevant labels.** Prefer \
artists documented on Klockworks, Ostgut Ton, Tresor, Innervisions, Rush Hour, \
Internasjonal, SUED, Livity Sound, Hemlock, or equivalent labels.

- **Use ONLY real, commercially released track titles you are certain exist.** \
The set is made playable by resolving each track on Deezer, so a fictional or \
misremembered title produces no audio. Before committing to a title, apply this \
test: "Has this been released on a real label, pressed on vinyl, or licensed to \
streaming platforms?" If the answer is anything less than "yes, I am certain", \
choose a different track by that artist instead. Prefer the artist's most \
well-known or most-played tracks. Never invent a title, never use a title you \
only vaguely remember. A set with eight real, playable tracks is far better \
than a set with invented titles that produce the wrong audio.

- Do not use em dashes or en dashes in the reasons. Use commas or full stops.

---

## Example 1, seed: "hypnotic Berlin techno for 2am, 6 tracks"

{{
  "title": "Klock Hour, 2am Berghain main floor",
  "tracks": [
    {{
      "artist": "Marcel Dettmann",
      "title": "Plain",
      "reason": "Opens with a stripped kick and hat, anchors a half-full floor without committing energy.",
      "energy": 4
    }},
    {{
      "artist": "Ben Klock",
      "title": "Subzero",
      "reason": "Subby low-end joins the kick, the room starts breathing in time.",
      "energy": 5
    }},
    {{
      "artist": "Stef Mendesidis",
      "title": "Skyline Drive",
      "reason": "Klockworks-aligned tension build, sets up the first peak without giving it away.",
      "energy": 6
    }},
    {{
      "artist": "DVS1",
      "title": "Klockworks 18",
      "reason": "Peak, relentless kick weight, a single-element track that holds the room locked.",
      "energy": 8
    }},
    {{
      "artist": "Norman Nodge",
      "title": "Tracking",
      "reason": "Sustains the peak with a different texture, avoids the drop-off after a single peak track.",
      "energy": 7
    }},
    {{
      "artist": "Function",
      "title": "Tide",
      "reason": "Mixes out, slower decay, hat-driven, hands off cleanly to whatever follows.",
      "energy": 6
    }}
  ]
}}

Arc: 4, 5, 6, 8, 7, 6 (warm-up, build, peak, sustain, transition out).

---

## Example 2, seed: "deep melodic warm-up set, Watergate vibe, 6 tracks"

{{
  "title": "Spree Warm-Up, Watergate Friday 23h",
  "tracks": [
    {{
      "artist": "Mr. Fingers",
      "title": "Mystery of Love",
      "reason": "Origin-of-deep-house chord stab, sets a generous, patient mood.",
      "energy": 2
    }},
    {{
      "artist": "Ame",
      "title": "Rej",
      "reason": "Innervisions melodic register, bass enters, the room begins to gather.",
      "energy": 3
    }},
    {{
      "artist": "Dixon",
      "title": "Where We At",
      "reason": "Classic Watergate sound, kicks pick up, swing finds the dancers.",
      "energy": 4
    }},
    {{
      "artist": "Stimming",
      "title": "Funkenflug",
      "reason": "Warmer textures, melodic motif over a steady groove, the room is committed.",
      "energy": 5
    }},
    {{
      "artist": "Tale Of Us",
      "title": "Like a Race",
      "reason": "Cinematic build, longer breakdowns, bridges into the next set cleanly.",
      "energy": 6
    }},
    {{
      "artist": "Henrik Schwarz",
      "title": "Walk Music",
      "reason": "Hand-off track, full energy, classical-electronic crossover that lets the headliner take over.",
      "energy": 7
    }}
  ]
}}

Arc: 2, 3, 4, 5, 6, 7 (steady patient build, appropriate for warm-up duty).

---

## Example 3, seed: "industrial peak-time techno, Tresor main floor, 6 tracks"

{{
  "title": "Globus Peak, Tresor Lower Level",
  "tracks": [
    {{
      "artist": "Surgeon",
      "title": "Force and Urgency",
      "reason": "Tresor stalwart, hard functional kick with industrial tension, commits the floor at the first peak.",
      "energy": 8
    }},
    {{
      "artist": "Paula Temple",
      "title": "Red Warning",
      "reason": "Noise-drenched industrial techno, distorted sub pressure, holds the room at peak intensity.",
      "energy": 9
    }},
    {{
      "artist": "Ancient Methods",
      "title": "The Jconstructor",
      "reason": "Saturated kick and feedback loops, Methkin label architect, peak saturation without a breakdown.",
      "energy": 9
    }},
    {{
      "artist": "Rebekah",
      "title": "Minus",
      "reason": "Harder modular texture, keeps the industrial pressure without repeating the exact palette.",
      "energy": 8
    }},
    {{
      "artist": "Regis",
      "title": "Gymnast",
      "reason": "British Murder Boys co-founder, stripped minimal industrial, first breath after the peak run.",
      "energy": 7
    }},
    {{
      "artist": "Blawan",
      "title": "Getting Me Down",
      "reason": "Ternesc label, rhythmic and percussive, transitions the floor out of full industrial into something breathable.",
      "energy": 6
    }}
  ]
}}

Arc: 8, 9, 9, 8, 7, 6 (sustained peak then controlled descent).

---

## Example 4, seed: "ambient closing set, 6am wind-down after a long night"

{{
  "title": "After the Floor, 6am Resolution",
  "tracks": [
    {{
      "artist": "Basic Channel",
      "title": "Phylyps Trak",
      "reason": "Chain Reaction dub techno pioneer, long reverb tails and minimal kick, the floor breathes out.",
      "energy": 5
    }},
    {{
      "artist": "Deepchord",
      "title": "Streamside",
      "reason": "Rod Modell atmospheric dub, textures replace rhythm, the crowd stops dancing and starts listening.",
      "energy": 4
    }},
    {{
      "artist": "Burial",
      "title": "Archangel",
      "reason": "Hyperdub, spectral 2-step ghost rhythm, emotional and still, the right track when the night is over.",
      "energy": 3
    }},
    {{
      "artist": "The Caretaker",
      "title": "All You Are Going to Want to Do Is Get Back There",
      "reason": "Haunted Ballroom series, decayed ballroom loops, the furthest point from a peak.",
      "energy": 2
    }},
    {{
      "artist": "Stars of the Lid",
      "title": "Requiem for Dying Mothers",
      "reason": "Kranky orchestral ambient, slow string swell, the final exhale of the night.",
      "energy": 2
    }},
    {{
      "artist": "William Basinski",
      "title": "Disintegration Loops 1.1",
      "reason": "Decaying tape loop, the room empties in silence. Nothing follows this.",
      "energy": 1
    }}
  ]
}}

Arc: 5, 4, 3, 2, 2, 1 (full descent to silence, resolution of the night).

---

## Now build the set

Seed (user's request): {seed}
Number of tracks: {n}

Return ONLY the JSON object. No explanation, no markdown fences, no \
preamble. The response must be parseable as JSON directly.
"""


def build_setlist_prompt(seed: str, n: int) -> str:
    """Return the legacy SETLIST_PROMPT filled with seed and track count."""
    return SETLIST_PROMPT.format(seed=seed, n=n)
