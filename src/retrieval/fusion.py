"""
fusion.py
---------
Reciprocal Rank Fusion (RRF): combines ranked result lists from BM25 and
vector search into a single ranking, without needing to normalize or
compare their raw scores directly (BM25 scores and cosine similarities
live on completely different scales, so naive score-averaging is
meaningless -- RRF sidesteps that by fusing on *rank position* instead).

score(doc) = sum over each ranking r that contains doc of  1 / (k + rank_in_r)

k=60 is the standard default from the original RRF paper (Cormack et al.,
2009) and works well without tuning in practice.
"""
from __future__ import annotations

from typing import NamedTuple

DEFAULT_RRF_K = 60


class FusedResult(NamedTuple):
    chunk_id: str
    score: float
    bm25_rank: int | None   # 1-indexed rank in the BM25 list, None if absent
    vector_rank: int | None  # 1-indexed rank in the vector list, None if absent


def reciprocal_rank_fusion(
    bm25_chunk_ids: list[str],
    vector_chunk_ids: list[str],
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
) -> list[FusedResult]:
    """
    bm25_chunk_ids / vector_chunk_ids: chunk IDs already sorted best-first
    by their respective retrievers. Only order matters here, not the
    original scores.
    """
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_chunk_ids)}
    vector_rank = {cid: i + 1 for i, cid in enumerate(vector_chunk_ids)}

    all_ids = set(bm25_chunk_ids) | set(vector_chunk_ids)
    fused = []
    for cid in all_ids:
        score = 0.0
        br = bm25_rank.get(cid)
        vr = vector_rank.get(cid)
        if br is not None:
            score += 1.0 / (k + br)
        if vr is not None:
            score += 1.0 / (k + vr)
        fused.append(FusedResult(chunk_id=cid, score=score, bm25_rank=br, vector_rank=vr))

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused[:top_k]
