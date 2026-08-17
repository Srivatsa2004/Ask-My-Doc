#!/usr/bin/env python3
"""
query.py
--------
Ad-hoc CLI to test hybrid retrieval against whatever indexes were built by
build_index.py. Prints each result's fused score, BM25/vector rank, and
citation metadata (doc, section, page) -- useful for sanity-checking
retrieval quality by eye before Day 3's reranker and Day 4's generation
step are wired in.

Usage:
    python scripts/query.py "what is the meal reimbursement limit" --fake-embedder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import DeterministicFakeEmbedder, OpenAIEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker, LocalOverlapReranker
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Query the hybrid retriever.")
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--db", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fake-embedder", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Apply Cohere cross-encoder reranking")
    parser.add_argument("--fake-reranker", action="store_true",
                         help="Use offline LocalOverlapReranker instead of Cohere (for dev/testing)")
    args = parser.parse_args()

    store = Store(db_path=Path(args.db)) if args.db else Store()

    if args.fake_embedder:
        embedder = DeterministicFakeEmbedder(dim=128)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set. Pass --fake-embedder for offline testing.", file=sys.stderr)
            sys.exit(1)
        embedder = OpenAIEmbedder()

    reranker = None
    if args.fake_reranker:
        reranker = LocalOverlapReranker()
    elif args.rerank:
        if not os.environ.get("COHERE_API_KEY"):
            print("Error: COHERE_API_KEY not set. Pass --fake-reranker for offline testing.", file=sys.stderr)
            sys.exit(1)
        reranker = CohereReranker()

    vector_store = QdrantVectorStore(dim=embedder.dim)
    bm25_index = BM25Index.load()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vector_store,
                                 bm25_index=bm25_index, reranker=reranker)

    results = retriever.retrieve(args.query, final_top_k=args.top_k)

    if not results:
        print("No results found. Did you run build_index.py first?")
        return

    print(f'\nQuery: "{args.query}"\n')
    for i, r in enumerate(results, 1):
        c = r.chunk
        loc = f"p.{c.page}" if c.page else "no page"
        score_line = f"fused_score={r.fused_score:.4f}  (bm25_rank={r.bm25_rank}, vector_rank={r.vector_rank})"
        if r.rerank_score is not None:
            score_line += f"  rerank_score={r.rerank_score:.4f}"
        print(f"[{i}] {score_line}")
        print(f"    source: {c.doc_id} | {c.section} | {loc} | chunk_id={c.chunk_id}")
        print(f"    text: {c.text[:200].strip()}...")
        print()

    vector_store.close()


if __name__ == "__main__":
    main()
