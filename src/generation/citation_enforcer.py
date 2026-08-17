"""
citation_enforcer.py
---------------------
Post-hoc verification layer that sits between the LLM's raw output and what
the user actually sees. Prompting a model to cite its sources (prompt.py)
is necessary but not sufficient -- models still occasionally cite a source
number that wasn't provided, or state a claim with no citation at all, or
cite a real source that doesn't actually say what the sentence claims. This
module catches all three and strips the offending content (per project
decision: strip unsupported claims rather than regenerate the whole answer).

Verification has two layers:
1. Citation validity: does the cited source number exist in what was
   actually retrieved? (cheap, exact)
2. Claim support (lightweight): does the cited source's text plausibly
   support the sentence, via lexical overlap? This is a heuristic, NOT a
   real entailment/NLI model -- it catches "cited source 1 but the claim
   is about something source 1 never mentions" (e.g. a fabricated number),
   but it can't catch subtler contradictions or logical inference errors.
   Swap in a real NLI model (e.g. roberta-large-mnli) or an LLM-as-judge
   call here for production-grade entailment checking; this heuristic
   exists to keep the pipeline dependency-light and network-free for a
   portfolio-scale project.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.prompt import NO_ANSWER_PHRASE
from src.retrieval.hybrid_retriever import RetrievedChunk

MIN_SUPPORT_OVERLAP = 0.15  # fraction of a sentence's content words that must appear in the cited source

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Split on whitespace that follows sentence-ending punctuation OR a citation marker's
# closing bracket, but only when the next sentence actually starts (capital letter,
# digit, or end of string). This deliberately does NOT split between a sentence and
# its own trailing citation marker(s) -- "...day. [1] Employees..." must split into
# "...day. [1]" and "Employees...", not strand "[1]" as its own fragment.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z0-9]|$)")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class DroppedClaim:
    sentence: str
    reason: str  # "invalid_citation" | "no_citation" | "unsupported_by_source"


@dataclass
class EnforcedAnswer:
    text: str                          # final, citation-verified answer shown to the user
    used_chunk_ids: list[str] = field(default_factory=list)   # chunks actually cited in the kept text
    dropped: list[DroppedClaim] = field(default_factory=list)
    is_no_answer: bool = False


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _claim_supported_by_source(sentence_no_citations: str, source_text: str) -> bool:
    claim_words = _tokenize(sentence_no_citations)
    if not claim_words:
        return False
    source_words = _tokenize(source_text)
    overlap = len(claim_words & source_words) / len(claim_words)
    return overlap >= MIN_SUPPORT_OVERLAP


def enforce_citations(raw_answer: str, source_map: dict[int, RetrievedChunk]) -> EnforcedAnswer:
    if raw_answer.strip() == NO_ANSWER_PHRASE or not source_map:
        return EnforcedAnswer(text=NO_ANSWER_PHRASE, is_no_answer=True)

    kept_sentences: list[str] = []
    used_chunk_ids: list[str] = []
    dropped: list[DroppedClaim] = []

    for sentence in _split_sentences(raw_answer):
        citation_numbers = [int(n) for n in _CITATION_RE.findall(sentence)]
        sentence_text_only = _CITATION_RE.sub("", sentence).strip()

        if not citation_numbers:
            dropped.append(DroppedClaim(sentence=sentence, reason="no_citation"))
            continue

        valid_and_supported: list[int] = []
        for n in citation_numbers:
            chunk = source_map.get(n)
            if chunk is None:
                continue  # invalid citation: source number wasn't actually provided
            if _claim_supported_by_source(sentence_text_only, chunk.chunk.text):
                valid_and_supported.append(n)

        if not valid_and_supported:
            invalid_ns = [n for n in citation_numbers if n not in source_map]
            reason = "invalid_citation" if invalid_ns else "unsupported_by_source"
            dropped.append(DroppedClaim(sentence=sentence, reason=reason))
            continue

        # Keep the sentence, but rewrite its citation markers to include only the
        # ones that actually verified -- a partially-hallucinated citation list
        # (e.g. "[1][99]") should surface as "[1]", not silently keep the bad one.
        rebuilt_citations = "".join(f"[{n}]" for n in valid_and_supported)
        kept_sentences.append(f"{sentence_text_only} {rebuilt_citations}")
        for n in valid_and_supported:
            used_chunk_ids.append(source_map[n].chunk.chunk_id)

    if not kept_sentences:
        return EnforcedAnswer(text=NO_ANSWER_PHRASE, dropped=dropped, is_no_answer=True)

    return EnforcedAnswer(
        text=" ".join(kept_sentences),
        used_chunk_ids=list(dict.fromkeys(used_chunk_ids)),  # de-dupe, preserve order
        dropped=dropped,
    )
