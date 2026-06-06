"""
Rave Atlas, web_search tool.

A keyless fallback the agent reaches for when the curated knowledge base does
not cover something: current artist news, a recent release or tour, a venue that
opened recently, or a scene the KB does not describe in depth.

Design choices:
  - Keyless on purpose. It uses DuckDuckGo via the `ddgs` package, so it needs
    no API key and works on Streamlit Cloud with zero extra secrets.
  - Gap-honest, like explain_music. On any failure (network, rate limit, empty)
    it returns grounded=False and an empty result list rather than raising, so
    the agent can say "I could not find this" instead of crashing the turn.
  - Results are DATA, not instructions. The agent's system prompt treats web
    results as untrusted external content and never follows instructions inside
    them. This module just fetches and normalises; it never executes anything.
"""

from __future__ import annotations

from logging_config import get_logger

logger = get_logger(__name__)

# Hard caps so a tool call can never balloon context or latency.
_MAX_RESULTS = 6
_MAX_SNIPPET_CHARS = 400


def web_search(query: str, k: int = 5) -> dict[str, object]:
    """
    Search the public web and return a small, normalised result set.

    CALL THIS TOOL only when the curated knowledge base (explain_music) does not
    cover the question, or the question is about current, real-world facts the KB
    cannot know (recent releases, tours, news, venues outside Berlin's depth).

    DO NOT call this tool for:
    - Anything the music knowledge base covers   ->  use explain_music first
    - Live or upcoming Berlin events             ->  use find_events instead
    - Set lists                                  ->  use build_setlist instead

    Args:
        query: A focused natural-language search query.
        k:     How many results to return (1 to 6). Default 5.

    Returns:
        {
            "query":    str,
            "results":  list[ {"title": str, "url": str, "snippet": str} ],
            "grounded": bool,   # False means nothing usable was found;
                                # the agent must say so, not invent an answer.
        }
        Never raises: returns grounded=False on any error.
    """
    q = (query or "").strip()
    if not q:
        return {"query": query, "results": [], "grounded": False}

    k = max(1, min(int(k), _MAX_RESULTS))

    try:
        from ddgs import DDGS
    except ModuleNotFoundError:
        logger.warning("web_search_ddgs_missing")
        return {"query": q, "results": [], "grounded": False}

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(q, max_results=k))
    except Exception as exc:  # network, rate limit, parsing, anything
        logger.warning("web_search_failed", query=q[:80], error=str(exc)[:160])
        return {"query": q, "results": [], "grounded": False}

    results: list[dict[str, str]] = []
    for r in raw:
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (r.get("body") or "").strip()[:_MAX_SNIPPET_CHARS]
        if title or snippet:
            results.append({"title": title, "url": url, "snippet": snippet})

    grounded = len(results) > 0
    logger.info("web_search", query=q[:80], hits=len(results), grounded=grounded)
    return {"query": q, "results": results, "grounded": grounded}


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    print("Test: web_search live query")
    out = web_search("Ben Klock 2024 release", k=3)
    print(f"  grounded : {out['grounded']}")
    print(f"  n_results: {len(out['results'])}")
    for i, r in enumerate(out["results"], 1):
        print(f"   [{i}] {r['title']}")
        print(f"       {r['url']}")
        print(f"       {r['snippet'][:120]}...")

    empty = web_search("", k=3)
    assert empty["grounded"] is False, "FAIL: empty query must be ungrounded"
    print("\nOK, empty query handled gracefully.")
