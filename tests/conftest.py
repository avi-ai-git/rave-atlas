"""Shared fixtures for the Rave Atlas test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_module_caches():
    """
    Clear all module-level caches before each test so tests are fully
    independent regardless of execution order.

    Clears:
    - tools.artists._ARTIST_CACHE  (in-memory artist enrichment cache)
    - tools.setlist._DEEZER_CACHE  (in-memory Deezer search cache)
    - llm_client._cache            (in-memory LLM response cache)
    - safety._last_messages        (per-session duplicate-detection state)
    """
    from tools import artists as _artists
    from tools import setlist as _setlist
    import llm_client
    import safety

    _artists._ARTIST_CACHE.clear()
    _setlist._DEEZER_CACHE.clear()
    llm_client._cache.clear()
    safety._last_messages.clear()
    yield
