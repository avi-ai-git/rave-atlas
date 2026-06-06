"""
Rave Atlas, find_club tool.

A deterministic lookup over the Berlin club registry (tools/club_registry.py).
Returns a club's official website, events page, address, and berlin.de detail
page exactly, with no embedding lottery.

Why this is a tool and not RAG:
    Retrieval ranks chunks by topical similarity, so a query like "Tresor's
    website" surfaces the chunk that is *most about Tresor*, which is the
    rich history/venue chunk, not the small chunk that happens to carry the
    URL. For an exact fact (a URL, an address) a keyed lookup is correct and
    a vector search is not. This mirrors the project's "right altitude per
    surface" principle: deterministic where the answer is a fact, RAG where
    the answer is synthesis.
"""

from __future__ import annotations

from logging_config import get_logger
from tools.club_registry import ALL_CLUBS, get_club

logger = get_logger(__name__)


def find_club(name: str) -> dict[str, object]:
    """
    Look up a Berlin club's official links and address by name.

    CALL THIS TOOL when the user asks for a specific Berlin club's official
    website, event page, address, or "where do I find what's on at X", for
    any of the ~70 registered Berlin venues. This returns authoritative
    official links (harvested from each club's berlin.de Senate-registry
    page), not a web guess.

    DO NOT call this for live event listings on dates (use find_events), or
    for scene/history/genre questions (use explain_music).

    Args:
        name: The club name, e.g. "Tresor", "Club der Visionaere", "Renate".
              Matching is case-insensitive and tolerates partial names.

    Returns:
        {
            "found": bool,
            "name": str,
            "address": str,
            "website": str | None, # official site root
            "events_url": str | None, # direct programme/events page
            "berlin_de": str | None, # authoritative berlin.de detail page
            "instagram": str | None,
            "note": str, # one-line description
            "suggestions": list[str], # close names when found=False
        }
        found=False means the club is not in the registry, say so and offer
        the suggestions rather than inventing a URL.
    """
    entry = get_club(name)
    if entry is None:
        needle = name.strip().lower()
        suggestions = [
            c.name for c in ALL_CLUBS
            if any(tok and tok in c.name.lower() for tok in needle.split())
        ][:5]
        logger.info("find_club_miss", query=name[:60], suggestions=len(suggestions))
        return {
            "found": False,
            "name": name,
            "address": "",
            "website": None,
            "events_url": None,
            "berlin_de": None,
            "instagram": None,
            "note": (
                "That venue is not in the Berlin club registry. It may be a "
                "collective, a one-off party, or spelled differently."
            ),
            "suggestions": suggestions,
        }

    logger.info("find_club_hit", name=entry.name, has_website=bool(entry.website))
    return {
        "found": True,
        "name": entry.name,
        "address": entry.address,
        "website": entry.website,
        "events_url": entry.events_url,
        "berlin_de": entry.berlin_de,
        "instagram": entry.instagram,
        "note": entry.note,
        "suggestions": [],
    }


if __name__ == "__main__":
    print("Test 1: exact hit, Tresor")
    r = find_club("Tresor")
    print(f" found={r['found']} website={r['website']} addr={r['address']}")
    assert r["found"] is True
    assert r["website"] == "https://tresorberlin.com"

    print("Test 2: fuzzy hit, 'visionaere'")
    r = find_club("visionaere")
    print(f" found={r['found']} name={r['name']} website={r['website']}")
    assert r["found"] is True
    assert "clubdervisionaere" in (r["website"] or "")

    print("Test 3: miss, 'Fabric London'")
    r = find_club("Fabric London")
    print(f" found={r['found']} suggestions={r['suggestions']}")
    assert r["found"] is False
    assert r["website"] is None

    print("Test 4: link-only club still resolves, 'Void Club'")
    r = find_club("Void Club")
    print(f" found={r['found']} berlin_de={bool(r['berlin_de'])}")
    assert r["found"] is True
    assert r["berlin_de"]

    print("All assertions passed.")
