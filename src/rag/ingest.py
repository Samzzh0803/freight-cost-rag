"""
Corpus ingest pipeline: .md files -> FAISS index + chunks.parquet

Chunking strategy: split on markdown heading boundaries (## / #) first to keep
headings attached to their body text. If a section exceeds MAX_CHUNK_CHARS,
split further on blank lines. This prevents tables being sliced mid-row and
headings being detached from content.

Embeddings are L2-normalised before indexing so that IndexFlatIP gives cosine
similarity (dot product on unit vectors = cosine). This lets retriever.py use
a cosine-similarity threshold for "no relevant chunk found" detection.
"""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

REQUIRED_FRONTMATTER_KEYS = {"source", "source_url", "date_retrieved", "corpus", "topic"}
MAX_CHUNK_CHARS = 1200  # generous cap — keeps tables and bullet blocks intact
MIN_CHUNK_CHARS = 80
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter (between --- delimiters) from body text."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError("Missing YAML frontmatter block (expected --- delimiters)")
    fm_raw, body = match.group(1), match.group(2)
    meta: dict = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip().strip('"')
    missing = REQUIRED_FRONTMATTER_KEYS - meta.keys()
    if missing:
        raise ValueError(f"Frontmatter missing required keys: {missing}")
    return meta, body.strip()


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split markdown body into semantic chunks.

    Strategy:
    1. Split on ## or # heading lines — each heading starts a new section.
    2. If a section is still > max_chars, split further on blank lines.
    3. Filter out chunks below MIN_CHUNK_CHARS.

    This keeps headings attached to their body and preserves table rows.
    """
    # Split into sections at each heading (## or #), keeping the heading line
    sections = re.split(r"(?=\n#{1,2} )", text)

    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Section too large: split on blank lines (paragraph boundaries)
            paragraphs = re.split(r"\n{2,}", section)
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current and len(current) + len(para) + 2 > max_chars:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = (current + "\n\n" + para).strip() if current else para
            if current.strip():
                chunks.append(current.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def ingest_corpus(corpus_dir: Path, index_dir: Path) -> None:
    """
    Walk corpus_dir for .md files -> parse -> chunk -> embed -> FAISS + parquet.

    Embeddings are L2-normalised; index is IndexFlatIP (cosine similarity).
    Retriever.py should use dot-product similarity (higher = more similar).

    Outputs:
      index_dir/faiss.index   -- FAISS IndexFlatIP on L2-normalised vectors
      index_dir/chunks.parquet -- chunk_id, text, source, source_url, corpus, topic, file
    """
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(corpus_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found under {corpus_dir}")

    records: list[dict] = []
    chunk_id = 0

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        try:
            meta, body = _parse_frontmatter(text)
        except ValueError as e:
            raise ValueError(f"{md_path}: {e}") from e

        for chunk_text in _chunk_text(body):
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": meta["source"],
                    "source_url": meta["source_url"],
                    "corpus": meta["corpus"],
                    "topic": meta["topic"],
                    "file": str(md_path.relative_to(corpus_dir)),
                }
            )
            chunk_id += 1

    if not records:
        raise RuntimeError("No chunks produced -- check corpus .md files")

    texts = [r["text"] for r in records]

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # L2-normalise so dot product = cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # inner product on normalised = cosine
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "faiss.index"))

    df = pd.DataFrame(records)
    df.to_parquet(index_dir / "chunks.parquet", index=False)

    print(f"Ingest complete: {len(records)} chunks from {len(md_files)} files -> {index_dir}")
    by_corpus = df.groupby("corpus").size().to_dict()
    for corpus, count in sorted(by_corpus.items()):
        print(f"  {corpus}: {count} chunks")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent
    ingest_corpus(
        corpus_dir=repo_root / "data" / "corpus",
        index_dir=repo_root / "data" / "index",
    )
