"""
Rave Atlas — knowledge base ingestion pipeline.

Loads every markdown file from knowledge_base/, chunks it into 200–400 word
pieces, attaches metadata (source, doc_type, genre), embeds with a local
sentence-transformers model, and persists to ChromaDB.

Run once after cloning, and again whenever knowledge_base/ content changes:
    uv run python ingest.py
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

import config
from logging_config import get_logger

logger = get_logger(__name__)

KB_DIR: Path = Path(__file__).parent / "knowledge_base"
EMBED_MODEL: str = "all-MiniLM-L6-v2"
COLLECTION_NAME: str = "rave_atlas_kb"

# Maps filename stem → (doc_type, genre | None)
_FILE_META: dict[str, tuple[str, str | None]] = {
    "genres_techno":         ("genre",   "techno"),
    "genres_house":          ("genre",   "house"),
    "genres_psytrance":      ("genre",   "psytrance"),
    "genres_dubstep":        ("genre",   "dubstep"),
    "berlin_scene_history":  ("history", None),
    "labels":                ("labels",  None),
    "track_anatomy":         ("theory",  None),
}


def _chunk_text(text: str, min_words: int = 150, max_words: int = 400) -> list[str]:
    """
    Split markdown text into chunks of min_words–max_words words.

    Splits on double newlines (paragraph boundaries) first, then merges
    short paragraphs until min_words is reached, and splits long ones at
    sentence boundaries. Strips markdown heading markers from output.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        # If adding this paragraph would exceed max_words, flush first
        if current_words and len(current_words) + len(words) > max_words:
            chunks.append(" ".join(current_words))
            current_words = []

        current_words.extend(words)

        # If we've hit min_words, flush
        if len(current_words) >= min_words:
            chunks.append(" ".join(current_words))
            current_words = []

    if current_words:
        chunks.append(" ".join(current_words))

    # Clean markdown heading syntax from chunks (## Heading → Heading)
    cleaned = [re.sub(r"^#+\s*", "", c, flags=re.MULTILINE) for c in chunks]
    return [c for c in cleaned if len(c.split()) >= 30]


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if absent."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest() -> int:
    """
    Ingest all markdown files from knowledge_base/ into ChromaDB.

    Returns the total number of chunks indexed.
    """
    collection = get_collection()
    total_chunks = 0

    # rglob so curated root files AND auto-enriched files under knowledge_base/
    # community/ (written by automation/kb_enrich.py) are both ingested.
    for md_file in sorted(KB_DIR.rglob("*.md")):
        stem = md_file.stem
        # community/ files are crowd-sourced; tag them so retrieval can tell
        # them apart from the curated, hand-written canon.
        if md_file.parent.name == "community":
            doc_type, genre = "community", None
        else:
            doc_type, genre = _FILE_META.get(stem, ("general", None))

        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{stem}_{i:03d}"
            ids.append(chunk_id)
            documents.append(chunk)
            meta: dict[str, str] = {
                "source": md_file.name,
                "doc_type": doc_type,
            }
            if genre:
                meta["genre"] = genre
            metadatas.append(meta)

        if ids:
            # Upsert so re-running ingest is idempotent
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(
                "ingested",
                file=md_file.name,
                doc_type=doc_type,
                genre=genre,
                chunks=len(ids),
            )
            total_chunks += len(ids)

    logger.info("ingest_complete", total_chunks=total_chunks)
    return total_chunks


if __name__ == "__main__":
    print("Running ingestion …")
    n = ingest()
    print(f"Indexed {n} chunks total.\n")

    # Test query
    collection = get_collection()
    results = collection.query(
        query_texts=["what BPM is techno played at"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print("Query: 'what BPM is techno played at'")
    print(f"Returned {len(results['documents'][0])} chunks:\n")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  source   : {meta.get('source')}")
        print(f"  doc_type : {meta.get('doc_type')}")
        print(f"  genre    : {meta.get('genre', 'n/a')}")
        print(f"  distance : {dist:.4f}")
        print(f"  text     : {doc[:120]}…")
        print()

    # Assertions
    doc_types = [m.get("doc_type") for m in results["metadatas"][0]]
    genres = [m.get("genre") for m in results["metadatas"][0]]

    assert "genre" in doc_types, "Expected at least one chunk with doc_type='genre'"
    assert "techno" in genres, "Expected at least one chunk with genre='techno'"

    print("All assertions passed.")
