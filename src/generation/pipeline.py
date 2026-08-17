"""
pipeline.py
-----------
The full "Ask My Docs" query path: retrieve relevant chunks, generate a
citation-tagged answer, verify/strip via citation_enforcer, and attach
human-readable source references (doc, section, page) for whatever
citations survived enforcement. This is the main entry point the API
(Day 6) will call.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generation.citation_enforcer import EnforcedAnswer, enforce_citations
from src.generation.generator import Generator
from src.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class SourceReference:
    chunk_id: str
    doc_id: str
    section: str | None
    page: int | None


@dataclass
class AskResult:
    query: str
    answer: str
    sources: list[SourceReference]
    is_no_answer: bool
    n_claims_dropped: int  # how many sentences the enforcer stripped -- a health signal for eval/monitoring


class AskPipeline:
    def __init__(self, retriever: HybridRetriever, generator: Generator, retrieve_top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.retrieve_top_k = retrieve_top_k

    def ask(self, query: str) -> AskResult:
        retrieved = self.retriever.retrieve(query, final_top_k=self.retrieve_top_k)

        if not retrieved:
            return AskResult(query=query, answer=EnforcedAnswer(text="", is_no_answer=True).text or
                              "I don't have enough information in the provided sources to answer this question.",
                              sources=[], is_no_answer=True, n_claims_dropped=0)

        raw_answer, source_map = self.generator.generate(query, retrieved)
        enforced = enforce_citations(raw_answer, source_map)

        # build source references only for chunks that actually survived enforcement --
        # never show a citation the user can't trust
        used_ids = set(enforced.used_chunk_ids)
        sources = [
            SourceReference(chunk_id=r.chunk.chunk_id, doc_id=r.chunk.doc_id,
                             section=r.chunk.section, page=r.chunk.page)
            for r in retrieved if r.chunk.chunk_id in used_ids
        ]
        # preserve citation order (order chunks were cited in the answer), not retrieval order
        sources.sort(key=lambda s: enforced.used_chunk_ids.index(s.chunk_id))

        return AskResult(
            query=query, answer=enforced.text, sources=sources,
            is_no_answer=enforced.is_no_answer, n_claims_dropped=len(enforced.dropped),
        )
