"""
Rave Atlas, knowledge base ingestion pipeline.

Loads every markdown file from knowledge_base/, parses YAML frontmatter
(doc_type, scope, genre, source), strips it from the body, chunks the body
into 150-400 word pieces, attaches the frontmatter as ChromaDB metadata,
embeds with a local sentence-transformers model, and persists to ChromaDB.

Frontmatter is the source of truth for metadata. A filename fallback
(_FILE_META) covers legacy files that predate frontmatter. Research dumps
(multi-file concatenations marked with "## FILE:") and build/process notes
(doc_type: build_notes / audit / plan) are skipped, never ingested.

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

# Fallback (doc_type, genre) for legacy files that lack YAML frontmatter.
# When frontmatter is present it always wins (see _resolve_meta).
_FILE_META: dict[str, tuple[str, str | None]] = {
    "genres_techno": ("genre", "techno"),
    "genres_house": ("genre", "house"),
    "genres_psytrance": ("genre", "psytrance"),
    "genres_dubstep": ("genre", "dubstep"),
    "berlin_scene_history": ("history", None),
    "labels": ("labels", None),
    "track_anatomy": ("theory", None),
}

# doc_type values that mark a file as build/process notes, not knowledge.
# These are never ingested even if they live under knowledge_base/.
_SKIP_DOC_TYPES: frozenset[str] = frozenset(
    {"build_notes", "audit", "plan", "meta", "todo", "notes"}
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    Split a leading YAML frontmatter block (``---`` ... ``---``) from the body.

    Returns (frontmatter_dict, body). If there is no frontmatter, returns
    ({}, original_text). Only flat ``key: value`` lines are parsed, which is
    all this knowledge base uses.
    """
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, flags=re.DOTALL)
    if not match:
        return {}, text

    block = match.group(1)
    body = text[match.end():]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                meta[key] = val
    return meta, body


def _derive_scope(stem: str, fm: dict[str, str]) -> str:
    """Resolve the retrieval scope: frontmatter wins, then filename heuristics."""
    if fm.get("scope"):
        return fm["scope"]
    if stem.startswith("city_primer_"):
        return "city:" + stem[len("city_primer_"):]
    if stem.startswith("berlin"):
        return "berlin"
    return "general"


def _resolve_meta(stem: str, fm: dict[str, str], filename: str) -> dict[str, str]:
    """
    Build the ChromaDB metadata for a file. Frontmatter-first, filename
    fallback. City primers are normalised to doc_type='city_primer' so the
    retrieval allowlist can target them uniformly regardless of how the
    individual file tagged itself ('city', 'city_primer', etc.).
    """
    fallback_dt, fallback_genre = _FILE_META.get(stem, ("general", None))

    if stem.startswith("city_primer_"):
        doc_type = "city_primer"
    else:
        doc_type = fm.get("doc_type") or fallback_dt

    genre = fm.get("genre") or fallback_genre
    scope = _derive_scope(stem, fm)
    source = fm.get("source") or filename

    meta: dict[str, str] = {
        "source": source,
        "doc_type": doc_type,
        "scope": scope,
    }
    # ChromaDB metadata values cannot be None; only add genre when meaningful.
    # "all" is a non-discriminating value, so it is not stored as a filter.
    if genre and genre.strip().lower() != "all":
        meta["genre"] = genre
    return meta


def _extract_title(body: str, stem: str) -> str:
    """Return the file's first H1 as its title, or a readable fallback from the stem."""
    for line in body.splitlines():
        m = re.match(r"^#\s+(.*)$", line.strip())
        if m and m.group(1).strip():
            return m.group(1).strip()
    # Fallback: turn "berlin_club_doors" into "Berlin club doors".
    return stem.replace("_", " ").strip().capitalize()


def _chunk_text(
    text: str,
    title: str = "",
    min_words: int = 150,
    max_words: int = 400,
    overlap_words: int = 40,
) -> list[str]:
    """
    Split markdown body text into retrieval chunks, heading-aware and overlapped.

    Two deliberate choices drive retrieval quality:

    1. Heading breadcrumb. Each chunk is prefixed with "Title > Section" drawn
       from the file's H1 and the nearest markdown heading. Without this, a
       chunk about a door policy loses the word "Berghain" if that lived only
       in the section heading, and the query "Berghain door policy" no longer
       matches it. The breadcrumb puts the topic words inside every chunk.

    2. Overlap. Consecutive chunks within a section share the last
       `overlap_words` words, so a fact split across a chunk boundary is still
       fully present in one chunk and stays retrievable.

    Chunks stay within min_words-max_words. Code-fence lines are dropped
    defensively, and fragments under 30 words are discarded as too thin to be
    useful context.
    """
    # Drop code-fence lines defensively (real KB files have none; this guards
    # against any stray ``` that would otherwise pollute a chunk).
    text = re.sub(r"^\s*```.*$", "", text, flags=re.MULTILINE)

    # Walk the body, grouping content under its nearest heading so each chunk
    # can carry that heading as context.
    sections: list[tuple[str, str]] = [] # (heading_text, section_body)
    current_heading = ""
    buf: list[str] = []

    def _flush_section() -> None:
        if any(line.strip() for line in buf):
            sections.append((current_heading, "\n".join(buf)))

    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)$", line.strip())
        if heading:
            _flush_section()
            buf = []
            current_heading = heading.group(1).strip()
        else:
            buf.append(line)
    _flush_section()

    chunks: list[str] = []
    for heading, body in sections:
        breadcrumb = " > ".join(p for p in (title, heading) if p)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]

        current_words: list[str] = []
        section_chunks: list[list[str]] = []

        def _emit(words: list[str]) -> list[str]:
            """Record a chunk and return the overlap tail to seed the next one."""
            section_chunks.append(words)
            return words[-overlap_words:] if overlap_words else []

        for para in paragraphs:
            words = para.split()
            if not words:
                continue
            if current_words and len(current_words) + len(words) > max_words:
                current_words = _emit(current_words)
            current_words.extend(words)
            if len(current_words) >= min_words:
                current_words = _emit(current_words)

        # Flush the tail, unless it is only the carried-over overlap.
        if current_words and (not section_chunks or len(current_words) > overlap_words):
            section_chunks.append(current_words)

        for words in section_chunks:
            body_text = " ".join(words)
            if len(words) < 30:
                continue
            chunks.append(f"{breadcrumb}\n{body_text}" if breadcrumb else body_text)

    return chunks


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
    Ingest all knowledge_base markdown into ChromaDB. Returns the total
    number of chunks indexed.
    """
    collection = get_collection()
    total_chunks = 0
    skipped: list[str] = []

    # rglob so root files AND auto-enriched files under knowledge_base/
    # community/ are both ingested.
    for md_file in sorted(KB_DIR.rglob("*.md")):
        stem = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)

        # Guard 1: research dumps (many files concatenated with "## FILE:")
        # would ingest as one noisy blob full of frontmatter/code-fence text.
        if "## FILE:" in body:
            skipped.append(f"{md_file.name} (research dump)")
            logger.warning("skipped_research_dump", file=md_file.name)
            continue

        # Guard 2: build/process notes (plans, audits) are not knowledge.
        if fm.get("doc_type", "").strip().lower() in _SKIP_DOC_TYPES:
            skipped.append(f"{md_file.name} (doc_type={fm.get('doc_type')})")
            logger.info("skipped_non_kb_doc", file=md_file.name, doc_type=fm.get("doc_type"))
            continue

        if md_file.parent.name == "community":
            meta_base = {
                "source": md_file.name,
                "doc_type": "community",
                "scope": "general",
            }
        else:
            meta_base = _resolve_meta(stem, fm, md_file.name)

        chunks = _chunk_text(body, title=_extract_title(body, stem))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for i, chunk in enumerate(chunks):
            ids.append(f"{stem}_{i:03d}")
            documents.append(chunk)
            metadatas.append(dict(meta_base)) # one copy per chunk

        if ids:
            # Upsert so re-running ingest is idempotent.
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(
                "ingested",
                file=md_file.name,
                doc_type=meta_base["doc_type"],
                scope=meta_base["scope"],
                genre=meta_base.get("genre"),
                chunks=len(ids),
            )
            total_chunks += len(ids)

    logger.info("ingest_complete", total_chunks=total_chunks, skipped=len(skipped))
    if skipped:
        logger.info("ingest_skipped_files", files=skipped)
    return total_chunks


if __name__ == "__main__":
    print("Running ingestion ...")
    n = ingest()
    print(f"Indexed {n} chunks total.\n")

    collection = get_collection()

    # Diverse test queries spanning the expanded KB.
    queries = [
        "what BPM is techno played at",
        "what is the Berghain door policy",
        "MDMA harm reduction and drug checking",
        "techno clubs and scene in Tbilisi",
    ]
    for q in queries:
        results = collection.query(
            query_texts=[q],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        print(f"Query: {q!r}")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            print(f" [{dist:.3f}] doc_type={meta.get('doc_type')} "
                  f"scope={meta.get('scope')} source={meta.get('source')}")
            print(f" {doc[:90]}...")
        print()

    # Assertions: metadata is populated and no frontmatter leaked into chunks.
    sample = collection.get(limit=200, include=["documents", "metadatas"])
    assert sample["metadatas"], "FAIL: no chunks ingested"
    for meta in sample["metadatas"]:
        assert meta.get("doc_type"), "FAIL: chunk missing doc_type"
        assert meta.get("scope"), "FAIL: chunk missing scope"
    for doc in sample["documents"]:
        head = doc.lstrip()[:40].lower()
        assert not head.startswith("doc_type:"), (
            "FAIL: frontmatter leaked into chunk text"
        )
        assert not head.startswith("---"), "FAIL: frontmatter fence leaked into chunk"

    # Scope coverage: we should have at least a berlin scope.
    scopes = {m.get("scope") for m in sample["metadatas"]}
    assert "berlin" in scopes, "FAIL: expected at least one berlin-scoped chunk"

    print(f"Scopes present: {sorted(scopes)}")
    print("All assertions passed.")
