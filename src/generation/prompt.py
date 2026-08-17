"""
prompt.py
---------
Builds the prompt for citation-constrained generation. Retrieved chunks are
presented to the model as numbered sources ([1], [2], ...) rather than by
their raw chunk_id -- short numeric citations are far more reliable for a
model to produce correctly and consistently than long string IDs, and the
number->chunk_id mapping (`source_map`) is exactly what citation_enforcer.py
needs to verify each citation the model emits.
"""
from __future__ import annotations

from src.retrieval.hybrid_retriever import RetrievedChunk

NO_ANSWER_PHRASE = "I don't have enough information in the provided sources to answer this question."

SYSTEM_PROMPT = f"""You are an enterprise knowledge assistant. Answer the user's question using ONLY the numbered sources provided below. Follow these rules exactly:

1. Every factual claim in your answer MUST end with a citation marker like [1] or [2] referencing the source number it came from. If a sentence draws on multiple sources, cite all of them, e.g. [1][3].
2. Do NOT use any knowledge beyond what is in the provided sources, even if you know the answer from elsewhere.
3. Do NOT cite a source number that isn't in the provided list.
4. If the sources don't contain enough information to answer the question, respond with EXACTLY this sentence and nothing else: "{NO_ANSWER_PHRASE}"
5. Be concise. Do not add commentary, caveats, or a references list -- citations belong inline as [N] markers only.
"""


def format_sources(retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
    """Returns (formatted source block, {source_number: RetrievedChunk}) for citation verification later."""
    source_map: dict[int, RetrievedChunk] = {}
    lines = []
    for i, r in enumerate(retrieved, start=1):
        source_map[i] = r
        loc = f", p.{r.chunk.page}" if r.chunk.page else ""
        section = f" ({r.chunk.section})" if r.chunk.section else ""
        lines.append(f"[{i}] Source: {r.chunk.doc_id}{section}{loc}\n{r.chunk.text}\n")
    return "\n".join(lines), source_map


def build_user_prompt(query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
    sources_block, source_map = format_sources(retrieved)
    user_prompt = f"""Sources:
{sources_block}

Question: {query}

Answer (remember: every claim needs a [N] citation, use ONLY the sources above):"""
    return user_prompt, source_map
