"""
Tests for the eval harness. Split deliberately into two tiers:

TIER 1 (always runs, no API keys needed) -- the actual CI gate. These check
things that are deterministic and don't require a real LLM's judgment:
retrieval recall, and whether citation enforcement actually blocks
hallucinated content from ever reaching the user. If either regresses,
the build should fail.

TIER 2 (skipped unless OPENAI_API_KEY, COHERE_API_KEY, and ANTHROPIC_API_KEY
are all set) -- real end-to-end quality against the golden set, including
whether the system correctly refuses to answer unanswerable questions. This
genuinely requires a real LLM's contextual judgment; see
src/generation/generator.py's FakeSmartGenerator docstring for why a lexical
heuristic can't reliably substitute for this on a small corpus. These tests
run locally with real keys, or in CI if the repo has the secrets configured
-- they are informational/soft in CI without secrets (skipped, not failed).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from eval.harness import load_golden_set, run_eval
from src.generation.generator import AnthropicGenerator, FakeHallucinatingGenerator, FakeSmartGenerator
from src.generation.pipeline import AskPipeline
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import DeterministicFakeEmbedder, OpenAIEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker, LocalOverlapReranker
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Store

REAL_KEYS_AVAILABLE = all(
    os.environ.get(k) for k in ("OPENAI_API_KEY", "COHERE_API_KEY", "ANTHROPIC_API_KEY")
)


def _offline_pipeline(generator, db_path) -> tuple[AskPipeline, QdrantVectorStore]:
    """Builds a pipeline against the project's real sample docs (data/raw/),
    ingested fresh into a temp DB, using offline fakes for embedding/reranking."""
    import sys as _sys
    sys_path_root = Path(__file__).resolve().parent.parent
    if str(sys_path_root) not in _sys.path:
        _sys.path.insert(0, str(sys_path_root))
    from scripts.ingest import ingest_file
    from src.ingestion.parsers import PARSER_BY_EXTENSION

    store = Store(db_path=db_path)
    raw_dir = sys_path_root / "data" / "raw"
    for filepath in raw_dir.iterdir():
        if filepath.suffix.lower() in PARSER_BY_EXTENSION:
            ingest_file(filepath, store, domain="enterprise_kb")

    embedder = DeterministicFakeEmbedder(dim=128)
    vstore = QdrantVectorStore(dim=embedder.dim, path=db_path.parent / "qdrant")
    reranker = LocalOverlapReranker()
    bm25 = BM25Index()

    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore,
                                 bm25_index=bm25, reranker=reranker)
    retriever.build_indexes()

    return AskPipeline(retriever=retriever, generator=generator), vstore


# ---------- TIER 1: hard CI gate, no API keys needed ----------

def test_retrieval_recall_is_perfect_on_golden_set(tmp_path):
    """Every positive (answerable) question in the golden set must retrieve
    at least one of its expected chunks. This is deterministic given the
    fixed sample documents and doesn't depend on generation quality at all
    -- a regression here means chunking, indexing, or fusion broke."""
    pipeline, vstore = _offline_pipeline(FakeSmartGenerator(), tmp_path / "test.db")
    try:
        report = run_eval(pipeline)
        assert report.retrieval_recall == 1.0, (
            f"retrieval recall dropped to {report.retrieval_recall:.1%} -- "
            f"check chunking/indexing for a regression"
        )
    finally:
        vstore.close()


def test_enforcement_blocks_all_hallucinations_on_golden_set(tmp_path):
    """The core citation-enforcement guarantee: run EVERY golden-set question
    through a generator that deliberately hallucinates on every single call,
    and confirm the fabricated content never survives into the final answer,
    for any question. This is the single most important test in the project
    -- it's what makes 'citations you can trust' true rather than aspirational."""
    pipeline, vstore = _offline_pipeline(FakeHallucinatingGenerator(), tmp_path / "test.db")
    try:
        golden_set = load_golden_set()
        for item in golden_set:
            result = pipeline.ask(item["question"])
            assert "10,000" not in result.answer, (
                f"{item['id']}: fabricated claim survived enforcement!"
            )
            assert "retroactively" not in result.answer, (
                f"{item['id']}: uncited claim survived enforcement!"
            )
    finally:
        vstore.close()


def test_golden_set_has_at_least_one_negative_example():
    golden_set = load_golden_set()
    negatives = [q for q in golden_set if not q["expected_chunk_ids"]]
    assert len(negatives) >= 1, "golden set must include at least one unanswerable question"


# ---------- TIER 2: real end-to-end quality, requires live API keys ----------

@pytest.mark.skipif(not REAL_KEYS_AVAILABLE, reason="requires OPENAI_API_KEY, COHERE_API_KEY, ANTHROPIC_API_KEY")
def test_real_pipeline_correctly_refuses_unanswerable_question(tmp_path):
    store = Store(db_path=tmp_path / "test.db")
    from scripts.ingest import ingest_file
    from src.ingestion.parsers import PARSER_BY_EXTENSION
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    for filepath in raw_dir.iterdir():
        if filepath.suffix.lower() in PARSER_BY_EXTENSION:
            ingest_file(filepath, store, domain="enterprise_kb")

    embedder = OpenAIEmbedder()
    vstore = QdrantVectorStore(dim=embedder.dim, path=tmp_path / "qdrant")
    reranker = CohereReranker()
    bm25 = BM25Index()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore,
                                 bm25_index=bm25, reranker=reranker)
    retriever.build_indexes()

    pipeline = AskPipeline(retriever=retriever, generator=AnthropicGenerator())
    try:
        result = pipeline.ask("What is the vacation carryover policy for employees?")
        assert result.is_no_answer, "real pipeline should decline to answer an unanswerable question"
    finally:
        vstore.close()


@pytest.mark.skipif(not REAL_KEYS_AVAILABLE, reason="requires OPENAI_API_KEY, COHERE_API_KEY, ANTHROPIC_API_KEY")
def test_real_pipeline_overall_pass_rate_meets_bar(tmp_path):
    store = Store(db_path=tmp_path / "test.db")
    from scripts.ingest import ingest_file
    from src.ingestion.parsers import PARSER_BY_EXTENSION
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    for filepath in raw_dir.iterdir():
        if filepath.suffix.lower() in PARSER_BY_EXTENSION:
            ingest_file(filepath, store, domain="enterprise_kb")

    embedder = OpenAIEmbedder()
    vstore = QdrantVectorStore(dim=embedder.dim, path=tmp_path / "qdrant")
    reranker = CohereReranker()
    bm25 = BM25Index()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore,
                                 bm25_index=bm25, reranker=reranker)
    retriever.build_indexes()

    pipeline = AskPipeline(retriever=retriever, generator=AnthropicGenerator())
    try:
        report = run_eval(pipeline)
        assert report.overall_pass_rate >= 0.9, f"pass rate {report.overall_pass_rate:.1%} below 90% bar"
    finally:
        vstore.close()
