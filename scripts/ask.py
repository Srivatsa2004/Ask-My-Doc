#!/usr/bin/env python3
"""
ask.py
------
End-to-end CLI: retrieve, generate a citation-tagged answer, verify/strip
via citation_enforcer, print the final answer with its surviving sources.

Usage:
    export OPENAI_API_KEY=sk-...
    export COHERE_API_KEY=...
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/ask.py "what is the meal reimbursement limit"

    # fully offline (no API keys, for testing the pipeline plumbing):
    python scripts/ask.py "what is the meal reimbursement limit" --fake-all
    # offline, deliberately hallucinating generator (to see enforcement catch it):
    python scripts/ask.py "what is the meal reimbursement limit" --fake-all --hallucinate
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.generator import AnthropicGenerator, FakeCitingGenerator, FakeHallucinatingGenerator, FakeSmartGenerator
from src.generation.pipeline import AskPipeline
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import DeterministicFakeEmbedder, OpenAIEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker, LocalOverlapReranker
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Ask a question end-to-end.")
    parser.add_argument("query")
    parser.add_argument("--db", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fake-all", action="store_true",
                         help="Use offline fakes for embedder, reranker, and generator (no API keys needed)")
    parser.add_argument("--hallucinate", action="store_true",
                         help="With --fake-all, use the deliberately-hallucinating fake generator")
    args = parser.parse_args()

    store = Store(db_path=Path(args.db)) if args.db else Store()

    if args.fake_all:
        embedder = DeterministicFakeEmbedder(dim=128)
        reranker = LocalOverlapReranker()
        generator = FakeHallucinatingGenerator() if args.hallucinate else FakeSmartGenerator()
    else:
        for var in ("OPENAI_API_KEY", "COHERE_API_KEY", "ANTHROPIC_API_KEY"):
            if not os.environ.get(var):
                print(f"Error: {var} not set. Pass --fake-all for offline testing.", file=sys.stderr)
                sys.exit(1)
        embedder = OpenAIEmbedder()
        reranker = CohereReranker()
        generator = AnthropicGenerator()

    vector_store = QdrantVectorStore(dim=embedder.dim)
    bm25_index = BM25Index.load()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vector_store,
                                 bm25_index=bm25_index, reranker=reranker)
    pipeline = AskPipeline(retriever=retriever, generator=generator, retrieve_top_k=args.top_k)

    result = pipeline.ask(args.query)

    print(f'\nQuery: "{result.query}"\n')
    print(f"Answer: {result.answer}\n")
    if result.n_claims_dropped:
        print(f"(Note: {result.n_claims_dropped} unsupported claim(s) were caught and removed by citation enforcement.)\n")
    if result.sources:
        print("Sources:")
        for s in result.sources:
            loc = f", p.{s.page}" if s.page else ""
            print(f"  - {s.doc_id} | {s.section}{loc} [{s.chunk_id}]")

    vector_store.close()


if __name__ == "__main__":
    main()
