"""
chunker.py
----------
Turns a list of `Section` objects (from parsers.py) into `Chunk` rows ready
for storage.py. Two things matter here for downstream citation quality:

1. Structure-aware splitting: chunks are packed from whole sections rather
   than a blind fixed-token sliding window across the whole document, so a
   chunk doesn't silently straddle two unrelated topics without record of it.
2. Page-exact boundaries for paginated formats (PDF): a chunk never spans
   two PDF pages, so `Chunk.page` is always exactly correct -- never "the
   first page of several this chunk actually covers." Sections without page
   info (DOCX/HTML) pack purely by token budget across headings instead.

No content is ever silently dropped: a section too small to stand alone is
merged into an adjacent chunk (same page, if page-bound) rather than discarded.
"""
from __future__ import annotations

import re

from src.ingestion.parsers import Section
from src.storage import Chunk, make_chunk_id

DEFAULT_CHUNK_TOKENS = 350
DEFAULT_OVERLAP_TOKENS = 60
MIN_CHUNK_TOKENS = 40  # below this, a trailing group is merged into the previous chunk, never dropped

# Token counting uses a regex word/punctuation tokenizer rather than a model-specific
# BPE tokenizer (e.g. tiktoken). This intentionally avoids a network dependency at
# ingestion time (tiktoken's encoding files are fetched remotely on first use) and gives
# a deterministic, offline-safe approximation. It runs ~10-15% higher than true BPE
# token counts for English text, which is fine since DEFAULT_CHUNK_TOKENS already has
# headroom below typical embedding/LLM context limits. Swap in tiktoken here if the
# deployment environment has network access and you want exact counts.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(_tokenize(text))


def _split_long_section(sec: Section, max_tokens: int, overlap_tokens: int) -> list[Section]:
    """Token-sliding-window split for a single section that's too long to be one chunk.
    Each piece keeps the parent section's heading/page so metadata is never lost."""
    tokens = _tokenize(sec.text)
    if len(tokens) <= max_tokens:
        return [sec]

    pieces = []
    start = 0
    step = max_tokens - overlap_tokens
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        piece_text = " ".join(tokens[start:end])
        pieces.append(Section(text=piece_text, heading=sec.heading, page=sec.page))
        if end == len(tokens):
            break
        start += step
    return pieces


def _label_headings(group: list[Section]) -> str | None:
    """Collapse the (possibly several) headings covered by a chunk into a display label."""
    seen: list[str] = []
    for s in group:
        if s.heading and (not seen or seen[-1] != s.heading):
            seen.append(s.heading)
    if not seen:
        return None
    if len(seen) == 1:
        return seen[0]
    return f"{seen[0]} \u2013 {seen[-1]}"  # en dash range, e.g. "1. Overview – 3. Limits"


def _label_page(group: list[Section]) -> int | None:
    pages = {s.page for s in group if s.page is not None}
    if not pages:
        return None
    # invariant (enforced by the packing loop below): a group never mixes pages
    assert len(pages) == 1, f"internal error: chunk group spans multiple pages {pages}"
    return pages.pop()


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    # Step 1: pre-split any section too long to fit in one chunk on its own.
    expanded: list[Section] = []
    for sec in sections:
        if count_tokens(sec.text) > max_tokens:
            expanded.extend(_split_long_section(sec, max_tokens, overlap_tokens))
        else:
            expanded.append(sec)

    # Step 2: greedily pack sections into groups. A group never spans two
    # different (non-None) page values -- that keeps PDF citations page-exact.
    # Sections with page=None (DOCX/HTML) pack purely by token budget.
    groups: list[list[Section]] = []
    current: list[Section] = []
    current_tokens = 0

    for sec in expanded:
        sec_tokens = count_tokens(sec.text)
        page_changed = (
            current and current[-1].page is not None
            and sec.page is not None and sec.page != current[-1].page
        )
        would_overflow = current and (current_tokens + sec_tokens > max_tokens)

        if current and (page_changed or would_overflow):
            groups.append(current)
            current, current_tokens = [], 0

        current.append(sec)
        current_tokens += sec_tokens

    if current:
        groups.append(current)

    # Step 3: merge an undersized trailing group into the previous one, but only
    # when they share a page (or neither has page info) -- never drop content,
    # and never violate the page-exactness invariant from step 2.
    merged: list[list[Section]] = []
    for group in groups:
        if (
            merged
            and sum(count_tokens(s.text) for s in group) < MIN_CHUNK_TOKENS
            and merged[-1][-1].page == group[0].page
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)

    # Step 4: materialize Chunk rows with accurate section/page labels.
    chunks: list[Chunk] = []
    char_cursor = 0
    for i, group in enumerate(merged):
        text = "\n\n".join(s.text for s in group)
        chunk_id = make_chunk_id(doc_id, i)
        chunks.append(Chunk(
            chunk_id=chunk_id, doc_id=doc_id, chunk_index=i,
            text=text, token_count=count_tokens(text),
            section=_label_headings(group), page=_label_page(group),
            char_start=char_cursor, char_end=char_cursor + len(text),
        ))
        char_cursor += len(text) + 1

    return chunks
