"""
storage.py
----------
SQLite-backed storage for documents and chunks.

This is the single source of truth for chunk metadata used throughout the
pipeline: retrieval (Day 2), reranking (Day 3), and citation enforcement
(Day 4) all resolve back to rows in the `chunks` table via `chunk_id`.

Schema design notes:
- `documents` tracks one row per source file (dedup by content hash).
- `chunks` stores the actual retrievable units. Every chunk carries enough
  metadata (page, section heading, char offsets) to render a human-readable
  citation like "policy.pdf, p.4, 'Refund Policy'" without re-parsing the
  source file.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "ask_my_docs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    filepath     TEXT NOT NULL,
    filetype     TEXT NOT NULL,           -- pdf | docx | html
    content_hash TEXT NOT NULL,
    domain       TEXT,
    ingested_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    chunk_index  INTEGER NOT NULL,        -- order within the document
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    section      TEXT,                    -- nearest heading, e.g. "3.2 Refund Policy"
    page         INTEGER,                 -- PDF page number (1-indexed), NULL for html/docx
    char_start   INTEGER,                 -- offset into the document's flattened text
    char_end     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


@dataclass
class Document:
    doc_id: str
    filename: str
    filepath: str
    filetype: str
    content_hash: str
    domain: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    section: Optional[str] = None
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


def compute_content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def make_doc_id(filename: str, content_hash: str) -> str:
    stem = Path(filename).stem.lower().replace(" ", "_")[:40]
    return f"{stem}_{content_hash[:8]}"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk{chunk_index:04d}"


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_document(self, doc: Document) -> bool:
        """Insert a document. Returns False if this content_hash already exists (dedup)."""
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO documents (doc_id, filename, filepath, filetype, content_hash, domain) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc.doc_id, doc.filename, doc.filepath, doc.filetype, doc.content_hash, doc.domain),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(chunk_id, doc_id, chunk_index, text, token_count, section, page, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.doc_id, c.chunk_index, c.text, c.token_count,
                     c.section, c.page, c.char_start, c.char_end)
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            return self._row_to_chunk(row) if row else None

    def all_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index").fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not row:
                return None
            return Document(**{k: row[k] for k in row.keys() if k != "ingested_at"})

    def stats(self) -> dict:
        with self._connect() as conn:
            n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return {"documents": n_docs, "chunks": n_chunks}

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], chunk_index=row["chunk_index"],
            text=row["text"], token_count=row["token_count"], section=row["section"],
            page=row["page"], char_start=row["char_start"], char_end=row["char_end"],
        )
