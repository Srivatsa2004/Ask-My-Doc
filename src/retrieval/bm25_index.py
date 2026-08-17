"""
bm25_index.py
-------------
Sparse lexical retrieval via BM25 (rank_bm25). Complements the vector
store: BM25 is strong on exact terms, acronyms, IDs, and rare words that
embeddings can blur together, which matters a lot for enterprise docs full
of policy codes, dollar figures, and proper nouns.

Persisted to disk via pickle so it doesn't need to be rebuilt from scratch
every process start -- rebuild explicitly via `scripts/build_index.py`
whenever the underlying chunks change.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import NamedTuple

from rank_bm25 import BM25Okapi

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Hit(NamedTuple):
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self.chunk_ids = chunk_ids
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[BM25Hit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called (or load()ed) before search()")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [BM25Hit(chunk_id=cid, score=float(s)) for cid, s in ranked[:top_k] if s > 0]

    def save(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self._bm25}, f)

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_PATH) -> "BM25Index":
        idx = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx.chunk_ids = data["chunk_ids"]
        idx._bm25 = data["bm25"]
        return idx
