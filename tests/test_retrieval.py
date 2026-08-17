"""
Tests for BM25, vector store, RRF fusion, and the combined HybridRetriever.

These run fully offline: Qdrant is used in its local/embedded mode (a real
Qdrant instance, just not networked), and DeterministicFakeEmbedder stands
in for OpenAI so no API key or network call is needed. This exercises real
indexing/search/fusion logic, not mocks of it -- only the embedding
*semantics* are fake, not the retrieval mechanics.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import Section
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import DeterministicFakeEmbedder
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Document, Store


# ---------- BM25 ----------

def test_bm25_finds_exact_term_match():
    idx = BM25Index()
    idx.build(
        chunk_ids=["a", "b", "c"],
        texts=[
            "Meal reimbursements are capped at $75 per day for domestic travel.",
            "Remote employees must attend all scheduled team meetings.",
            "The onboarding buddy program pairs new hires with a teammate.",
        ],
    )
    hits = idx.search("meal reimbursement cap", top_k=3)
    assert hits, "expected at least one BM25 hit"
    assert hits[0].chunk_id == "a", "exact-term chunk should rank first"


def test_bm25_save_and_load_roundtrip():
    # Note: uses 3 docs, not 2. rank_bm25's classic (non-"+1"-smoothed) IDF formula
    # -- log(N-n+0.5) - log(n+0.5) -- evaluates to exactly 0 whenever a term appears
    # in precisely half of an even-sized corpus, which zeroes out its score entirely.
    # That's a real quirk of tiny toy corpora, not something worth working around in
    # BM25Index itself (any real chunk set will have far more than 2 documents).
    idx = BM25Index()
    idx.build(chunk_ids=["x", "y", "z"], texts=["alpha beta gamma", "delta epsilon zeta", "eta theta iota"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "idx.pkl"
        idx.save(path)
        loaded = BM25Index.load(path)
    hits = loaded.search("alpha", top_k=3)
    assert hits[0].chunk_id == "x"


# ---------- Fusion ----------

def test_rrf_favors_items_ranked_high_in_both_lists():
    bm25 = ["doc1", "doc2", "doc3"]
    vector = ["doc2", "doc1", "doc4"]
    fused = reciprocal_rank_fusion(bm25, vector, top_k=4)
    fused_ids = [f.chunk_id for f in fused]
    # doc1 and doc2 both appear near the top of both lists -> should outrank doc3/doc4
    assert set(fused_ids[:2]) == {"doc1", "doc2"}


def test_rrf_includes_items_present_in_only_one_list():
    fused = reciprocal_rank_fusion(bm25_chunk_ids=["only_bm25"], vector_chunk_ids=["only_vector"], top_k=5)
    ids = {f.chunk_id for f in fused}
    assert ids == {"only_bm25", "only_vector"}


# ---------- Vector store (real embedded Qdrant, fake embeddings) ----------

@pytest.fixture
def tmp_vector_store():
    tmpdir = tempfile.mkdtemp()
    embedder = DeterministicFakeEmbedder(dim=32)
    store = QdrantVectorStore(dim=embedder.dim, path=Path(tmpdir) / "qdrant")
    yield store, embedder
    store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_vector_store_upsert_and_search_roundtrip(tmp_vector_store):
    vstore, embedder = tmp_vector_store
    texts = ["the cat sat on the mat", "quarterly revenue grew 12 percent"]
    ids = ["c1", "c2"]
    vectors = embedder.embed(texts)
    vstore.upsert(ids, vectors)
    assert vstore.count() == 2

    # searching with the exact same (fake) embedding for c1's text should return c1 first
    hits = vstore.search(embedder.embed_one(texts[0]), top_k=2)
    assert hits[0].chunk_id == "c1"


def test_vector_store_reupsert_overwrites_not_duplicates(tmp_vector_store):
    vstore, embedder = tmp_vector_store
    vec = embedder.embed_one("some text")
    vstore.upsert(["c1"], [vec])
    vstore.upsert(["c1"], [vec])  # re-upsert same chunk_id
    assert vstore.count() == 1


# ---------- End-to-end HybridRetriever ----------

@pytest.fixture
def populated_retriever():
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
    }

    for doc_id, sections in docs.items():
        store.upsert_document(Document(
            doc_id=doc_id, filename=f"{doc_id}.pdf", filepath="/fake", filetype="pdf", content_hash=doc_id,
        ))
        chunks = chunk_sections(sections, doc_id=doc_id)
        store.insert_chunks(chunks)

    embedder = DeterministicFakeEmbedder(dim=32)
    vstore = QdrantVectorStore(dim=embedder.dim, path=Path(tmpdir) / "qdrant")
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore)
    retriever.build_indexes()

    yield retriever
    vstore.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_hybrid_retriever_builds_indexes(populated_retriever):
    # Both sections within each fixture doc share a page/have no page info and fit
    # comfortably under the token budget, so the chunker correctly merges each doc's
    # 2 sections into 1 chunk -- 2 docs -> 2 chunks. See test_chunker.py for dedicated
    # coverage of when/why the chunker does or doesn't merge sections.
    assert populated_retriever.vector_store.count() == 2


def test_hybrid_retriever_bm25_surfaces_lexical_match(populated_retriever):
    # BM25 should reliably surface the exact-term match ("$75", "domestic travel")
    # even though the fake embedder can't provide real semantic signal.
    results = populated_retriever.retrieve("meal reimbursement domestic travel cap", fused_top_k=4)
    assert results, "expected non-empty results"
    top_chunk_texts = [r.chunk.text for r in results]
    assert any("$75" in t for t in top_chunk_texts), "expected the reimbursement-limits chunk to be retrieved"


def test_retrieved_chunk_carries_citation_metadata(populated_retriever):
    results = populated_retriever.retrieve("remote work security encryption", fused_top_k=4)
    assert results
    r = results[0]
    assert r.chunk.chunk_id
    assert r.chunk.doc_id
    # section/page metadata must survive the full round trip for citation rendering later
    assert r.chunk.section is not None
