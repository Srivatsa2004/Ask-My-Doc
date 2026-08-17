"""
Tests for reranking. Uses LocalOverlapReranker (offline, deterministic)
rather than CohereReranker so these run without a network call / API key --
same rationale as DeterministicFakeEmbedder in test_retrieval.py.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import Section
from src.retrieval.embeddings import DeterministicFakeEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import LocalOverlapReranker, RerankResult
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Document, Store


def test_local_reranker_ranks_higher_overlap_first():
    reranker = LocalOverlapReranker()
    query = "meal reimbursement domestic travel limit"
    docs = [
        "The onboarding buddy program pairs new hires with a teammate.",   # no overlap
        "Meal reimbursements are capped at $75 per day for domestic travel.",  # high overlap
    ]
    results = reranker.rerank(query, docs, top_n=2)
    assert results[0].index == 1, "the high-overlap document should rank first"
    assert results[0].relevance_score > results[1].relevance_score


def test_local_reranker_respects_top_n():
    reranker = LocalOverlapReranker()
    docs = ["a b c", "a b", "a", "b c d", "z"]
    results = reranker.rerank("a b c", docs, top_n=2)
    assert len(results) == 2


def test_local_reranker_handles_empty_documents():
    reranker = LocalOverlapReranker()
    assert reranker.rerank("query", [], top_n=5) == []


# ---------- End-to-end: hybrid retrieval + reranking ----------

@pytest.fixture
def populated_retriever_with_reranker():
    tmpdir = tempfile.mkdtemp()
    db_path = Path(tmpdir) / "test.db"
    store = Store(db_path=db_path)

    docs = {
        "expense_policy": [
            Section(text="Meal reimbursements are capped at $75 per day for domestic travel "
                          "and $100 per day for international travel.", heading="Reimbursement Limits", page=1),
            Section(text="Alcohol not related to client entertainment is not eligible for reimbursement.",
                    heading="Non-Reimbursable Items", page=1),
        ],
        "remote_policy": [
            Section(text="Remote employees must be available during core hours of 10am to 3pm.",
                    heading="Working Hours", page=None),
            Section(text="All remote work must be performed on company-issued devices with full-disk encryption.",
                    heading="Security Requirements", page=None),
        ],
        "onboarding": [
            Section(text="New hires complete mandatory compliance training covering data privacy "
                          "and security awareness during their first week.", heading="Week 1 Training", page=None),
        ],
    }

    for doc_id, sections in docs.items():
        store.upsert_document(Document(
            doc_id=doc_id, filename=f"{doc_id}.pdf", filepath="/fake", filetype="pdf", content_hash=doc_id,
        ))
        chunks = chunk_sections(sections, doc_id=doc_id)
        store.insert_chunks(chunks)

    embedder = DeterministicFakeEmbedder(dim=32)
    vstore = QdrantVectorStore(dim=embedder.dim, path=Path(tmpdir) / "qdrant")
    reranker = LocalOverlapReranker()
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore, reranker=reranker)
    retriever.build_indexes()

    yield retriever
    vstore.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_reranker_improves_final_ordering_over_noisy_fusion(populated_retriever_with_reranker):
    # The fake embedder makes vector search noise, which can drag irrelevant chunks
    # into the fused top results. The reranker (even the crude local one) should
    # still correctly promote the clearly-relevant chunk to #1 in the final result.
    results = populated_retriever_with_reranker.retrieve(
        "what is the meal reimbursement limit for domestic travel", final_top_k=3,
    )
    assert results
    assert "$75" in results[0].chunk.text
    assert results[0].rerank_score is not None


def test_retriever_without_reranker_still_works(populated_retriever_with_reranker):
    populated_retriever_with_reranker.reranker = None
    results = populated_retriever_with_reranker.retrieve("security encryption devices", final_top_k=3)
    assert results
    assert all(r.rerank_score is None for r in results)


# ---------- Golden eval set sanity checks ----------

def test_golden_set_is_well_formed():
    import json
    from pathlib import Path
    golden_set = json.loads((Path(__file__).resolve().parent.parent / "eval" / "golden_set.json").read_text())
    assert len(golden_set) >= 20, "golden set should have at least 20 questions per the project plan"
    ids = [q["id"] for q in golden_set]
    assert len(ids) == len(set(ids)), "golden set question IDs must be unique"
    for q in golden_set:
        assert q["question"].strip(), f"{q['id']}: question text must not be empty"
        assert "expected_chunk_ids" in q
        assert "reference_answer" in q
    negative_examples = [q for q in golden_set if not q["expected_chunk_ids"]]
    assert len(negative_examples) >= 1, "golden set should include at least one unanswerable/negative example"
