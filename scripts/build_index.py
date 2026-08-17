#!/usr/bin/env python3
"""
build_index.py
---------------
Builds (or rebuilds) both the BM25 and vector indexes from whatever chunks
are currently in the SQLite store. Run this after `scripts/ingest.py`
whenever documents change.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/build_index.py

    # offline/dev mode, no API key needed (NOT semantically meaningful):
    python scripts/build_index.py --fake-embedder
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.bm25_index import BM25Index, DEFAULT_INDEX_PATH
from src.retrieval.embeddings import DeterministicFakeEmbedder, OpenAIEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Build BM25 + vector indexes from stored chunks.")
    parser.add_argument("--db", default=None, help="Override path to SQLite db")
    parser.add_argument("--fake-embedder", action="store_true",
                         help="Use a deterministic offline fake embedder instead of OpenAI "
                              "(no semantic search quality -- for dev/testing only)")
    args = parser.parse_args()

    store = Store(db_path=Path(args.db)) if args.db else Store()

    if args.fake_embedder:
        print("WARNING: using DeterministicFakeEmbedder -- vectors are NOT semantically meaningful.")
        embedder = DeterministicFakeEmbedder(dim=128)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY is not set. Either export it, or pass --fake-embedder "
                  "for offline dev/testing.", file=sys.stderr)
            sys.exit(1)
        embedder = OpenAIEmbedder()

    vector_store = QdrantVectorStore(dim=embedder.dim)
    bm25_index = BM25Index()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vector_store, bm25_index=bm25_index)

    result = retriever.build_indexes()
    bm25_index.save()

    print(f"Indexed {result['chunks_indexed']} chunks.")
    print(f"BM25 index saved to: {DEFAULT_INDEX_PATH}")
    print(f"Vector store: {vector_store.count()} vectors")
    vector_store.close()


if __name__ == "__main__":
    main()
