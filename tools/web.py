"""
Rave Atlas, web_search tool.

A fallback the agent reaches for when the curated knowledge base does not cover
something: current artist news, a recent release or tour, a venue that opened
recently, or a scene the KB does not describe in depth.

Design choices:
  - Brave Search when BRAVE_SEARCH_API_KEY is set (SLA, 2 k free calls/month),
    DuckDuckGo otherwise (keyless, works on Streamlit Cloud with zero extra secrets).
  - Gap-honest, like explain_music. On any failure it returns grounded=False and
    an empty result list rather than raising, so the agent can say "I could not
    find this" instead of crashing the turn.
  - Results are DATA, not instructions. The agent's system prompt treats web
    results as untrusted external content and never follows instructions inside
    them. This module just fetches and normalises; it never executes anything.
"""

from __future__ import annotations

import config
from logging_config import get_logger

logger = get_logger(__name__)

_MAX_RESULTS = 6
_MAX_SNIPPET_CHARS = 400
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def _brave(query: str, k: int) -> list[dict[str, str]] | None:
    """Call Brave Search API. Returns None if key missing or request fails."""
    api_key = config.BRAVE_SEARCH_API_KEY
    if not api_key:
        return None
    try:
        import requests
        resp = requests.get(
            _BRAVE_ENDPOINT,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": k},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("brave_search_failed", query=query[:80], error=str(exc)[:160])
        return None

    results: list[dict[str, str]] = []
    for r in (data.get("web", {}).get("results") or []):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("description") or "").strip()[:_MAX_SNIPPET_CHARS]
        if title or snippet:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results or None


def _ddgs(query: str, k: int) -> list[dict[str, str]] | None:
    """DuckDuckGo keyless fallback. Returns None on any failure."""
    try:
        from ddgs import DDGS
    except ModuleNotFoundError:
        logger.warning("web_search_ddgs_missing")
        return None
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=k))
    except Exception as exc:
        logger.warning("ddgs_search_failed", query=query[:80], error=str(exc)[:160])
        return None

    results: list[dict[str, str]] = []
    for r in raw:
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (r.get("body") or "").strip()[:_MAX_SNIPPET_CHARS]
        if title or snippet:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results or None


def web_search(query: str, k: int = 5) -> dict[str, object]:
    """
    Search the public web and return a small, normalised result set.

    CALL THIS TOOL only when the curated knowledge base (explain_music) does not
    cover the question, or the question is about current, real-world facts the KB
    cannot know (recent releases, tours, news, venues outside Berlin's depth).

    DO NOT call this tool for:
    - Anything the music knowledge base covers -> use explain_music first
    - Live or upcoming Berlin events -> use find_events instead
    - Set lists -> use build_setlist instead

    Args:
        query: A focused natural-language search query.
        k: How many results to return (1 to 6). Default 5.

    Returns:
        {
            "query": str,
            "results": list[ {"title": str, "url": str, "snippet": str} ],
            "grounded": bool, # False means nothing usable was found;
                                # the agent must say so, not invent an answer.
        }
        Never raises: returns grounded=False on any error.
    """
    q = (query or "").strip()
    if not q:
        return {"query": query, "results": [], "grounded": False}

    k = max(1, min(int(k), _MAX_RESULTS))

    provider = "brave" if config.BRAVE_SEARCH_API_KEY else "ddgs"
    results = _brave(q, k) if config.BRAVE_SEARCH_API_KEY else _ddgs(q, k)

    # If Brave failed (rate limit, network), try DuckDuckGo as a last resort.
    if results is None and config.BRAVE_SEARCH_API_KEY:
        provider = "ddgs_fallback"
        results = _ddgs(q, k)

    results = results or []
    grounded = len(results) > 0
    logger.info("web_search", query=q[:80], provider=provider, hits=len(results), grounded=grounded)
    return {"query": q, "results": results, "grounded": grounded}


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    print(f"Provider: {'Brave' if config.BRAVE_SEARCH_API_KEY else 'DuckDuckGo (keyless)'}")
    print("Test: web_search live query")
    out = web_search("Ben Klock 2024 release", k=3)
    print(f" grounded : {out['grounded']}")
    print(f" n_results: {len(out['results'])}")
    for i, r in enumerate(out["results"], 1):
        print(f" [{i}] {r['title']}")
        print(f"     {r['url']}")
        print(f"     {r['snippet'][:120]}...")

    empty = web_search("", k=3)
    assert empty["grounded"] is False, "FAIL: empty query must be ungrounded"
    print("\nOK, empty query handled gracefully.")
