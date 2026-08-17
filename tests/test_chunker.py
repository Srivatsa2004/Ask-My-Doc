"""
Tests for the chunker. These specifically pin down two bugs found during
manual testing on Day 1:
  1. A chunk must never silently drop content (small trailing sections
     were previously discarded if below MIN_CHUNK_TOKENS).
  2. A chunk must never span two different PDF pages, and its recorded
     `page`/`section` metadata must be exactly correct for the text it
     actually contains -- not just inherited from the first sub-section.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_sections, count_tokens
from src.ingestion.parsers import Section


def all_chunk_text(chunks) -> str:
    return " ".join(c.text for c in chunks)


def test_no_content_is_dropped():
    sections = [
        Section(text="A " * 300, heading="Intro", page=1),
        Section(text="tiny trailing note", heading="Fine Print", page=1),  # well under MIN_CHUNK_TOKENS
    ]
    chunks = chunk_sections(sections, doc_id="doc1", max_tokens=350)
    combined = all_chunk_text(chunks)
    assert "tiny trailing note" in combined, "small trailing section must be merged, not dropped"


def test_chunk_never_spans_two_pages():
    sections = [
        Section(text="Page one content here.", heading="Sec A", page=1),
        Section(text="Page two content here.", heading="Sec B", page=2),
    ]
    chunks = chunk_sections(sections, doc_id="doc1", max_tokens=350)
    pages_per_chunk = {c.chunk_id: c.page for c in chunks}
    assert len(chunks) == 2, "different pages must not be packed into one chunk"
    assert pages_per_chunk[chunks[0].chunk_id] == 1
    assert pages_per_chunk[chunks[1].chunk_id] == 2


def test_multi_heading_chunk_gets_range_label():
    sections = [
        Section(text="short a", heading="1. Overview", page=1),
        Section(text="short b", heading="2. Details", page=1),
    ]
    chunks = chunk_sections(sections, doc_id="doc1", max_tokens=350)
    assert len(chunks) == 1
    assert chunks[0].section == "1. Overview \u2013 2. Details"


def test_long_section_splits_with_overlap_and_keeps_metadata():
    long_text = " ".join(f"word{i}" for i in range(1000))
    sections = [Section(text=long_text, heading="Big Section", page=5)]
    chunks = chunk_sections(sections, doc_id="doc1", max_tokens=350, overlap_tokens=60)
    assert len(chunks) > 1, "a section longer than max_tokens must be split"
    for c in chunks:
        assert c.page == 5
        assert c.section == "Big Section"
        assert c.token_count <= 350 + 5  # small slack for join spacing


def test_char_offsets_are_monotonic_and_non_overlapping_in_storage_order():
    sections = [
        Section(text="first bit of text", heading="A", page=1),
        Section(text="second bit", heading="B", page=2),
    ]
    chunks = chunk_sections(sections, doc_id="doc1", max_tokens=350)
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.char_start >= prev.char_end


def test_count_tokens_is_deterministic():
    assert count_tokens("hello, world!") == count_tokens("hello, world!")
    assert count_tokens("") == 0
