"""
reranker.py
------------
Cross-encoder reranking: takes the fused BM25+vector candidates (which are
ranked using two independent, comparatively weak signals) and rescoresthem
with a model that looks at the (query, document) pair jointly -- much more
accurate at judging "does this chunk actually answer this question" than
either BM25 or cosine similarity alone.

Two implementations behind the same interface:
- CohereReranker: real implementation via the Cohere Rerank API. Requires
  COHERE_API_KEY.
- LocalOverlapReranker: offline fallback for dev/testing without a network
  call. Scores by token-overlap (Jaccard-ish) between query and document --
  NOT a real cross-encoder, no contextual/semantic understanding, but
  deterministic and good enough to exercise the reranking plumbing and
  verify obviously-relevant chunks outrank obviously-irrelevant ones.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import NamedTuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class RerankResult(NamedTuple):
    index: int          # index into the original `documents` list passed in
    relevance_score: float


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        """Return up to top_n results, best-first, each referencing the original document index."""
        ...


class CohereReranker(Reranker):
    """Real reranker backed by the Cohere Rerank API. Requires COHERE_API_KEY."""

    def __init__(self, model: str = "rerank-v3.5"):
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Export it before using CohereReranker, "
                "e.g. `export COHERE_API_KEY=...`. For offline dev/testing, use "
                "LocalOverlapReranker instead."
            )
        import cohere
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        resp = self.client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [RerankResult(index=r.index, relevance_score=r.relevance_score) for r in resp.results]


class LocalOverlapReranker(Reranker):
    """
    Offline-safe stand-in for CohereReranker. Scores each document by Jaccard
    token overlap with the query. This has no semantic/contextual understanding
    (e.g. can't tell "car" relates to "automobile") -- it exists purely to test
    reranking plumbing without a network call. Swap in CohereReranker for any
    real quality evaluation.
    """

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        q_tokens = _tokenize(query)
        scored = []
        for i, doc in enumerate(documents):
            d_tokens = _tokenize(doc)
            if not q_tokens or not d_tokens:
                score = 0.0
            else:
                intersection = len(q_tokens & d_tokens)
                union = len(q_tokens | d_tokens)
                score = intersection / union if union else 0.0
            scored.append(RerankResult(index=i, relevance_score=score))
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored[:top_n]
