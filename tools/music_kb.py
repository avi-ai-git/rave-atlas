"""
Rave Atlas, explain_music RAG tool.

Retrieves grounded context from the curated electronic music knowledge base
and returns it for the agent to synthesise into a natural-language answer.
Gap-honesty: returns grounded=False rather than hallucinating when the query
falls outside the KB.
"""

from __future__ import annotations

from langsmith import traceable

from ingest import get_collection
from logging_config import get_logger

logger = get_logger(__name__)

# Cosine distance above which we consider the query out-of-scope for the KB.
# Calibrated against the corpus: on-topic queries return a best distance of
# 0.33 to 0.47; off-topic queries (geography, general knowledge) land above 0.70.
SIMILARITY_THRESHOLD: float = 0.65


@traceable(
    run_type="retriever",
    name="explain_music",
    metadata={"tab": "learn_the_scene", "component": "music_kb_tool"},
)
def explain_music(
    query: str,
    allowed_doc_types: list[str] | None = None,
    k: int = 4,
) -> dict[str, object]:
    """
    Retrieve grounded context from the Rave Atlas music knowledge base.

    CALL THIS TOOL when the user asks about:
    - Electronic music genres: techno, house, psytrance, dubstep, ambient, etc.
    - BPM ranges, rhythmic signatures, how to recognise a genre by ear
    - Berlin's electronic music scene, history, venues, culture
    - Record labels: Tresor, Ostgut Ton, BPitch Control, Innervisions, Klockworks
    - Artists' genre lineage or background (not real-time tour / release data)
    - How a dance music track is structured; DJ techniques; the energy arc
    - Iconic hardware: Roland TR-909, TB-303, and their role in electronic music

    DO NOT call this tool for:
    - Live or upcoming Berlin events → use find_events instead
    - Set-list or track recommendations → use build_setlist instead
    - Real-time artist releases or tour dates → use enrich_artist instead

    Args:
        query: The user's question in natural language.
        allowed_doc_types: Allowlist of doc_type values to restrict retrieval
            scope. Valid values: "genre", "history", "labels", "theory",
            "music_theory", "culture", "etiquette", "venue", "harm_reduction",
            "general_education". If None the entire KB is searched. Prefer
            narrowing when the question is clearly within one category.
        k: Number of chunks to retrieve (default 4).

    Returns:
        {
            "context": str, joined text of retrieved chunks, or a gap message,
            "sources": list[str], filenames that contributed chunks,
            "grounded": bool, False means the answer is not in the KB;
                               the agent must not invent an answer in this case.
        }
    """
    try:
        collection = get_collection()
    except Exception as exc:
        logger.error("chroma_unavailable", error=str(exc))
        return {
            "context": (
                "The music knowledge base is temporarily unavailable. "
                "Please try again in a moment."
            ),
            "sources": [],
            "grounded": False,
        }

    where_filter: dict | None = None
    if allowed_doc_types:
        where_filter = {"doc_type": {"$in": allowed_doc_types}}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("chroma_query_failed", query=query[:80], error=str(exc))
        return {
            "context": "Knowledge base query failed, please rephrase your question.",
            "sources": [],
            "grounded": False,
        }

    docs: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    if not docs:
        logger.info("explain_music_empty", query=query[:80])
        return {
            "context": (
                "No relevant content was found in the music knowledge base for this query."
            ),
            "sources": [],
            "grounded": False,
        }

    best_distance = min(distances)
    grounded = best_distance < SIMILARITY_THRESHOLD

    logger.info(
        "explain_music",
        query=query[:80],
        hits=len(docs),
        best_distance=round(best_distance, 4),
        grounded=grounded,
        allowed_doc_types=allowed_doc_types,
    )

    if not grounded:
        return {
            "context": (
                "This question is outside the Rave Atlas music knowledge base. "
                "I can answer questions about electronic music genres, Berlin scene "
                "history, record labels, and track structure, but not this topic."
            ),
            "sources": [],
            "grounded": False,
        }

    sources: list[str] = list(
        {m.get("source", "") for m in metas if m.get("source")}
    )
    context = "\n\n---\n\n".join(docs)

    return {
        "context": context,
        "sources": sources,
        "grounded": True,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: on-topic query, techno track structure")
    print("=" * 60)
    r1 = explain_music("how is a techno track structured")
    print(f" grounded : {r1['grounded']}")
    print(f" sources : {r1['sources']}")
    print(f" context : {r1['context'][:300]}...\n")
    assert r1["grounded"] is True, "FAIL: expected grounded=True for on-topic query"
    assert len(r1["sources"]) > 0, "FAIL: expected at least one source"
    assert len(r1["context"]) > 50, "FAIL: context should not be empty"

    print("=" * 60)
    print("Test 2: off-topic query, geography question")
    print("=" * 60)
    r2 = explain_music("what is the capital of France")
    print(f" grounded : {r2['grounded']}")
    print(f" context : {r2['context']}\n")
    assert r2["grounded"] is False, "FAIL: expected grounded=False for off-topic query"
    assert r2["sources"] == [], "FAIL: expected empty sources for ungrounded result"

    print("=" * 60)
    print("Test 3: allowlist filter, genre only")
    print("=" * 60)
    r3 = explain_music(
        "what defines the Berlin techno sound",
        allowed_doc_types=["genre"],
    )
    print(f" grounded : {r3['grounded']}")
    print(f" sources : {r3['sources']}")
    if r3["grounded"]:
        for src in r3["sources"]:
            assert src.startswith("genres_"), (
                f"FAIL: allowlist filter leaked non-genre source: {src}"
            )
    print()

    print("All assertions passed.")
