"""
embeddings.py
-------------
Defines the Embedder interface used by the vector store, plus two
implementations:

- OpenAIEmbedder: the real implementation, used in actual deployment.
  Requires OPENAI_API_KEY. Batches requests to stay within API limits.
- DeterministicFakeEmbedder: a hash-based, zero-dependency embedder used
  ONLY in tests and local dev when no API key / network is available. It
  is NOT semantically meaningful (it can't tell "dog" is close to "puppy")
  -- it exists purely so retrieval plumbing (indexing, search, fusion) can
  be exercised deterministically without hitting a network. Swap in
  OpenAIEmbedder for anything that needs real semantic search quality.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

EMBEDDING_DIM_OPENAI_SMALL = 1536


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(Embedder):
    """Real embedder backed by the OpenAI API. Requires OPENAI_API_KEY env var."""

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        from openai import OpenAI  # imported lazily so the fake embedder needs no openai key present

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before using OpenAIEmbedder, "
                "e.g. `export OPENAI_API_KEY=sk-...`. For offline dev/testing, use "
                "DeterministicFakeEmbedder instead."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dim = EMBEDDING_DIM_OPENAI_SMALL if model == "text-embedding-3-small" else 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        return vectors


class DeterministicFakeEmbedder(Embedder):
    """
    Offline-safe stand-in for tests and dev without network access. Produces
    a deterministic pseudo-random unit vector per text by seeding from its
    SHA-256 hash. Same text -> same vector, every run. NOT semantically
    meaningful -- do not use for anything but exercising retrieval plumbing.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # expand the 32-byte digest deterministically to `dim` floats in [-1, 1]
        vals = []
        seed = digest
        while len(vals) < self.dim:
            seed = hashlib.sha256(seed).digest()
            vals.extend((b / 127.5 - 1.0) for b in seed)
        vals = vals[: self.dim]
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]
