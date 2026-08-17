"""
Tests for citation enforcement. These are the most important tests in the
project: citation_enforcer.py is the mechanism that makes the "citations
you can trust" claim true rather than aspirational. Constructing source_map
by hand (rather than going through the full retrieval pipeline) here so
each test isolates exactly one enforcement behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.citation_enforcer import enforce_citations
from src.generation.generator import FakeCitingGenerator, FakeHallucinatingGenerator
from src.generation.pipeline import AskPipeline
from src.generation.prompt import NO_ANSWER_PHRASE, build_user_prompt
from src.ingestion.chunker import chunk_sections
from src.ingestion.parsers import Section
from src.retrieval.embeddings import DeterministicFakeEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from src.retrieval.reranker import LocalOverlapReranker
from src.retrieval.vector_store import QdrantVectorStore
from src.storage import Chunk, Document, Store


def make_retrieved_chunk(chunk_id: str, doc_id: str, text: str, section=None, page=None) -> RetrievedChunk:
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, chunk_index=0, text=text, token_count=len(text.split()),
                  section=section, page=page)
    return RetrievedChunk(chunk=chunk, fused_score=1.0, bm25_rank=1, vector_rank=1)


# ---------- Core enforcement behaviors ----------

def test_correctly_cited_sentence_is_kept():
    source_map = {1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day.")}
    answer = "Meal reimbursements are capped at $75 per day. [1]"
    result = enforce_citations(answer, source_map)
    assert not result.is_no_answer
    assert "c1" in result.used_chunk_ids
    assert "$75" in result.text
    assert not result.dropped


def test_citation_to_nonexistent_source_is_stripped():
    source_map = {1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day.")}
    answer = "Employees get an annual bonus of $10,000. [99]"
    result = enforce_citations(answer, source_map)
    assert "10,000" not in result.text
    assert result.dropped[0].reason == "invalid_citation"


def test_sentence_with_no_citation_is_stripped():
    source_map = {1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day.")}
    answer = "This policy applies retroactively to all prior years."
    result = enforce_citations(answer, source_map)
    assert result.is_no_answer  # nothing survives -> falls back to the no-answer phrase
    assert result.dropped[0].reason == "no_citation"


def test_partial_hallucination_keeps_good_sentence_strips_bad_one():
    source_map = {
        1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day for domestic travel."),
    }
    answer = (
        "Meal reimbursements are capped at $75 per day for domestic travel. [1] "
        "Employees also receive an annual bonus of $10,000. [99] "
        "This policy applies retroactively to all prior years."
    )
    result = enforce_citations(answer, source_map)
    assert "$75" in result.text
    assert "10,000" not in result.text
    assert "retroactively" not in result.text
    assert result.used_chunk_ids == ["c1"]
    assert len(result.dropped) == 2
    reasons = {d.reason for d in result.dropped}
    assert reasons == {"invalid_citation", "no_citation"}


def test_citation_to_real_source_but_unsupported_claim_is_stripped():
    # cites a real source number, but the claim's content has nothing to do with
    # what that source actually says -- this is the case the overlap heuristic exists for
    source_map = {1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day.")}
    answer = "The company headquarters relocated to a new building in 2024. [1]"
    result = enforce_citations(answer, source_map)
    assert result.is_no_answer
    assert result.dropped[0].reason == "unsupported_by_source"


def test_multi_citation_sentence_keeps_only_valid_ones():
    source_map = {
        1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day for domestic travel."),
    }
    answer = "Meal reimbursements are capped at $75 per day for domestic travel. [1][99]"
    result = enforce_citations(answer, source_map)
    assert "[1]" in result.text
    assert "[99]" not in result.text
    assert result.used_chunk_ids == ["c1"]


def test_no_answer_phrase_passed_through_untouched():
    source_map = {1: make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day.")}
    result = enforce_citations(NO_ANSWER_PHRASE, source_map)
    assert result.is_no_answer
    assert result.text == NO_ANSWER_PHRASE


def test_empty_source_map_forces_no_answer():
    result = enforce_citations("Anything the model says here. [1]", source_map={})
    assert result.is_no_answer


# ---------- Fake generators sanity-checked against real enforcement ----------

def test_fake_citing_generator_produces_fully_kept_answer():
    retrieved = [make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day for domestic travel.")]
    gen = FakeCitingGenerator()
    raw, source_map = gen.generate("meal cap?", retrieved)
    result = enforce_citations(raw, source_map)
    assert not result.is_no_answer
    assert result.used_chunk_ids == ["c1"]


def test_fake_hallucinating_generator_gets_caught_by_enforcement():
    retrieved = [make_retrieved_chunk("c1", "doc1", "Meal reimbursements are capped at $75 per day for domestic travel.")]
    gen = FakeHallucinatingGenerator()
    raw, source_map = gen.generate("meal cap?", retrieved)
    result = enforce_citations(raw, source_map)
    # the good sentence should survive; the fabricated bonus and uncited retroactivity claim should not
    assert "$75" in result.text
    assert "10,000" not in result.text
    assert "retroactively" not in result.text
    assert len(result.dropped) == 2


# ---------- Full pipeline, end to end, offline ----------

def _build_pipeline(tmp_path, generator):
    store = Store(db_path=tmp_path / "test.db")
    sections = [
        Section(text="Meal reimbursements are capped at $75 per day for domestic travel "
                      "and $100 per day for international travel.", heading="Reimbursement Limits", page=1),
    ]
    store.upsert_document(Document(doc_id="expense_policy", filename="expense_policy.pdf",
                                    filepath="/fake", filetype="pdf", content_hash="hash1"))
    store.insert_chunks(chunk_sections(sections, doc_id="expense_policy"))

    embedder = DeterministicFakeEmbedder(dim=32)
    vstore = QdrantVectorStore(dim=embedder.dim, path=tmp_path / "qdrant")
    retriever = HybridRetriever(store=store, embedder=embedder, vector_store=vstore, reranker=LocalOverlapReranker())
    retriever.build_indexes()

    return AskPipeline(retriever=retriever, generator=generator), vstore


def test_pipeline_end_to_end_with_citing_generator(tmp_path):
    pipeline, vstore = _build_pipeline(tmp_path, FakeCitingGenerator())
    try:
        result = pipeline.ask("what is the meal reimbursement limit for domestic travel")
        assert not result.is_no_answer
        assert result.sources, "expected at least one surviving source reference"
        assert result.sources[0].doc_id == "expense_policy"
        assert result.sources[0].page == 1
    finally:
        vstore.close()


def test_pipeline_end_to_end_with_hallucinating_generator(tmp_path):
    pipeline, vstore = _build_pipeline(tmp_path, FakeHallucinatingGenerator())
    try:
        result = pipeline.ask("what is the meal reimbursement limit for domestic travel")
        assert "10,000" not in result.answer
        assert "retroactively" not in result.answer
        assert result.n_claims_dropped == 2
        # the one legitimately-cited claim should still have made it through
        assert result.sources
    finally:
        vstore.close()
