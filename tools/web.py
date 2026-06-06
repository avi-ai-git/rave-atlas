"""
Rave Atlas, web_search tool.

A fallback the agent reaches for when the curated knowledge base does not cover
something: current artist news, a recent release or tour, a venue that opened
recently, or a scene the KB does not describe in depth.

Provider priority (first available key wins, DuckDuckGo always last):
  1. Serper (serper.dev) -- Google-backed, strongest index, set SERPER_API_KEY.
  2. Brave Search         -- strong independent index, set BRAVE_SEARCH_API_KEY.
  3. DuckDuckGo           -- keyless fallback, no SLA but always available.

Design choices:
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
_SERPER_ENDPOINT = "https://google.serper.dev/search"


def _serper(query: str, k: int) -> list[dict[str, str]] | None:
    """Call Serper (Google-backed). Returns None if key missing or request fails."""
    api_key = config.SERPER_API_KEY
    if not api_key:
        return None
    try:
        import requests
        resp = requests.post(
            _SERPER_ENDPOINT,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": k},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("serper_search_failed", query=query[:80], error=str(exc)[:160])
        return None

    results: list[dict[str, str]] = []
    for r in (data.get("organic") or []):
        title = (r.get("title") or "").strip()
        url = (r.get("link") or "").strip()
        snippet = (r.get("snippet") or "").strip()[:_MAX_SNIPPET_CHARS]
        if title or snippet:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results or None


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

    # Try providers in priority order: Serper > Brave > DuckDuckGo.
    provider, results = "none", None
    if config.SERPER_API_KEY:
        results = _serper(q, k)
        provider = "serper"
    if results is None and config.BRAVE_SEARCH_API_KEY:
        results = _brave(q, k)
        provider = "brave"
    if results is None:
        results = _ddgs(q, k)
        provider = "ddgs"

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

    if config.SERPER_API_KEY:
        active = "Serper (Google)"
    elif config.BRAVE_SEARCH_API_KEY:
        active = "Brave Search"
    else:
        active = "DuckDuckGo (keyless)"
    print(f"Provider: {active}")

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
