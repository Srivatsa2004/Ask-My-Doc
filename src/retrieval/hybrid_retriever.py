"""
hybrid_retriever.py
--------------------
Orchestrates the full retrieval path: build both indexes from stored
chunks, and at query time run BM25 + vector search in parallel, fuse with
RRF, then (optionally) rerank the fused candidates with a cross-encoder.
This is the main entry point Day 4's generation step will call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import Embedder
from src.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.storage import Chunk, Store


@dataclass
class RetrievedChunk:
    chunk: Chunk
    fused_score: float
    bm25_rank: Optional[int]
    vector_rank: Optional[int]
    rerank_score: Optional[float] = None  # set only when a reranker was used


class HybridRetriever:
    def __init__(self, store: Store, embedder: Embedder, vector_store, bm25_index: Optional[BM25Index] = None,
                 reranker: Optional[Reranker] = None):
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = reranker

    def build_indexes(self) -> dict:
        """(Re)build both BM25 and vector indexes from everything currently in the store."""
        chunks = self.store.all_chunks()
        if not chunks:
            return {"chunks_indexed": 0}

        chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]

        self.bm25_index.build(chunk_ids, texts)

        vectors = self.embedder.embed(texts)
        self.vector_store.upsert(chunk_ids, vectors)

        return {"chunks_indexed": len(chunks)}

    def retrieve(
        self,
        query: str,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        fused_top_k: int = 20,
        final_top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve candidates via hybrid search, then narrow to `final_top_k`.
        If a reranker is configured, the fused candidates (up to `fused_top_k`
        of them) are rescored with it and the final ordering comes from the
        reranker, not the fusion score -- the whole point of reranking is that
        it's a stronger, query-aware signal than fusion rank position alone.
        Without a reranker, `final_top_k` just truncates the fused list.
        """
        bm25_hits = self.bm25_index.search(query, top_k=bm25_top_k)
        query_vector = self.embedder.embed_one(query)
        vector_hits = self.vector_store.search(query_vector, top_k=vector_top_k)

        fused: list[FusedResult] = reciprocal_rank_fusion(
            bm25_chunk_ids=[h.chunk_id for h in bm25_hits],
            vector_chunk_ids=[h.chunk_id for h in vector_hits],
            top_k=fused_top_k,
        )

        candidates = []
        for f in fused:
            chunk = self.store.get_chunk(f.chunk_id)
            if chunk is None:
                continue  # stale index entry pointing at a since-deleted chunk; skip rather than crash
            candidates.append(RetrievedChunk(
                chunk=chunk, fused_score=f.score, bm25_rank=f.bm25_rank, vector_rank=f.vector_rank,
            ))

        if not self.reranker or not candidates:
            return candidates[:final_top_k]

        rerank_results = self.reranker.rerank(
            query=query, documents=[c.chunk.text for c in candidates], top_n=final_top_k,
        )
        reordered = []
        for r in rerank_results:
            rc = candidates[r.index]
            rc.rerank_score = r.relevance_score
            reordered.append(rc)
        return reordered
