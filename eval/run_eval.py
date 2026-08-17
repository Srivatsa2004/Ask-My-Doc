#!/usr/bin/env python3
"""
run_eval.py
-----------
Runs the golden set through the full pipeline (retrieve -> generate ->
enforce) and prints a report. This is what CI runs (via
tests/test_eval_harness.py, which wraps the same harness in assertions) --
this script is the human-readable version for local use.

Usage:
    python eval/run_eval.py --fake-all
    python eval/run_eval.py --fake-all --hallucinate   # stress-test with a bad generator
    python eval/run_eval.py                             # real OpenAI + Cohere + Anthropic
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import run_eval
from src.generation.generator import AnthropicGenerator, FakeCitingGenerator, FakeHallucinatingGenerator, FakeSmartGenerator
from src.generation.pipeline import AskPipeline
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import DeterministicFakeEmbedder, OpenAIEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker, LocalOverlapReranker
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Run the full eval harness against the golden set.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fake-all", action="store_true")
    parser.add_argument("--hallucinate", action="store_true", help="With --fake-all, stress-test with a bad generator")
    parser.add_argument("--fail-under", type=float, default=None,
                         help="Exit non-zero if overall_pass_rate falls below this fraction (for CI use)")
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

    report = run_eval(pipeline)

    print(f"{'ID':<5} {'Verdict':<8} {'Type':<10} Question")
    print("-" * 90)
    for r in report.results:
        verdict = "PASS" if r.correct_behavior else "FAIL"
        qtype = "negative" if r.is_negative_example else "positive"
        print(f"{r.id:<5} {verdict:<8} {qtype:<10} {r.question}")

    print(f"\nRetrieval recall (positive examples): {report.retrieval_recall:.1%}")
    print(f"Correct refusal rate (negative examples): {report.correct_refusal_rate:.1%}")
    print(f"False refusal rate (positive examples wrongly declined): {report.false_refusal_rate:.1%}")
    print(f"Avg claims dropped per answered question: {report.avg_claims_dropped:.2f}")
    print(f"Overall pass rate: {report.overall_pass_rate:.1%}")

    vector_store.close()

    if args.fail_under is not None and report.overall_pass_rate < args.fail_under:
        print(f"\nFAIL: overall pass rate {report.overall_pass_rate:.1%} is below threshold {args.fail_under:.1%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
