"""
generator.py
------------
Generator interface for the LLM call, plus:
- AnthropicGenerator: real implementation via the Anthropic SDK. Requires
  ANTHROPIC_API_KEY.
- FakeCitingGenerator: offline, deterministic, always cites everything it's
  given regardless of relevance. Used to test the enforcement "happy path"
  in isolation, NOT to test refusal behavior (see FakeSmartGenerator for that).
- FakeHallucinatingGenerator: offline, deterministic. Deliberately emits a
  claim with a citation number that doesn't exist in the source list, and
  a claim with no citation at all. This exists specifically to prove the
  enforcement layer actually strips unsupported content rather than just
  trusting whatever the model outputs -- untested enforcement code is not
  meaningfully different from no enforcement at all.
- FakeSmartGenerator: offline, deterministic, but checks query/source
  overlap before answering and refuses (emits NO_ANSWER_PHRASE) when
  nothing retrieved is actually relevant. Exists because FakeCitingGenerator
  will confidently "answer" ANY question by citing whatever chunks got
  retrieved, which makes it useless for testing refusal behavior in CI
  without a real API key -- see eval/harness.py's correct_refusal_rate.
  Still not a real language model: no paraphrasing, no multi-hop reasoning,
  no synthesis across sentences.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from src.generation.prompt import NO_ANSWER_PHRASE, SYSTEM_PROMPT, build_user_prompt
from src.retrieval.hybrid_retriever import RetrievedChunk

_WORD_RE = re.compile(r"[a-z0-9]+")

# Deliberately small, hand-picked stopword list -- not exhaustive NLP-grade
# stopword removal, just enough to stop generic query words ("what", "the",
# "employees", "policy") from padding out overlap scores and making every
# chunk look relevant. Content words (the actual topic of the question) are
# what should drive the relevance signal.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who", "whom",
    "how", "many", "much", "do", "does", "did", "have", "has", "had", "to", "of",
    "for", "in", "on", "at", "by", "with", "and", "or", "but", "if", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "they", "we", "my",
    "your", "their", "our", "as", "be", "being", "been", "from", "about",
}


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _content_words(text: str) -> set[str]:
    """Tokenize and drop stopwords -- used for relevance/overlap scoring so
    generic words don't make every chunk look relevant to every query."""
    return _tokenize(text) - _STOPWORDS


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
        """Returns (raw_answer_text, source_map) where source_map maps citation
        number -> RetrievedChunk, needed by citation_enforcer.py to verify claims."""
        ...


class AnthropicGenerator(Generator):
    """Real generator backed by the Anthropic API. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using AnthropicGenerator, "
                "e.g. `export ANTHROPIC_API_KEY=sk-ant-...`. For offline dev/testing, use "
                "FakeCitingGenerator or FakeHallucinatingGenerator instead."
            )
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
        user_prompt, source_map = build_user_prompt(query, retrieved)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return text, source_map


class FakeCitingGenerator(Generator):
    """Offline stand-in that always cites correctly -- exercises the happy path."""

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
        _, source_map = build_user_prompt(query, retrieved)
        if not source_map:
            return NO_ANSWER_PHRASE, source_map
        sentences = []
        for i, r in retrieved_items(source_map):
            first_sentence = r.chunk.text.strip().split(".")[0].strip()
            if first_sentence:
                sentences.append(f"{first_sentence}. [{i}]")
        return " ".join(sentences), source_map


class FakeHallucinatingGenerator(Generator):
    """
    Offline stand-in that deliberately misbehaves in two ways real LLMs
    sometimes do, to prove citation_enforcer.py catches both:
      1. Cites a source number that doesn't exist (e.g. [99]).
      2. States a claim with no citation marker at all.
    A correctly-cited sentence is also included, to confirm enforcement
    strips only the bad content and keeps the good.
    """

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
        _, source_map = build_user_prompt(query, retrieved)
        if not source_map:
            return NO_ANSWER_PHRASE, source_map
        first_idx, first_chunk = next(retrieved_items(source_map))
        good_sentence = f"{first_chunk.chunk.text.strip().split('.')[0].strip()}. [{first_idx}]"
        fabricated_out_of_range = "Employees also receive an annual bonus of $10,000. [99]"
        uncited_claim = "This policy applies retroactively to all prior years."
        return f"{good_sentence} {fabricated_out_of_range} {uncited_claim}", source_map


def retrieved_items(source_map: dict[int, RetrievedChunk]):
    return iter(sorted(source_map.items()))


class FakeSmartGenerator(Generator):
    """See module docstring. Uses the RELEVANCE SCORE the retriever already
    computed (rerank_score if a reranker was used, else fused_score) to decide
    whether to refuse -- rather than reinventing a second, weaker relevance
    signal inside the generator.

    IMPORTANT, HONEST LIMITATION (found via calibration against the golden
    set, see eval/golden_set.json and tests/test_eval_harness.py): on this
    project's small demo corpus, there is NO single score threshold that
    cleanly separates answerable from unanswerable questions. The one
    negative example in the golden set scores 0.0784, which falls *in the
    middle* of the positive examples' score range (0.0377-0.12) -- some
    genuinely answerable questions score lower than the one genuinely
    unanswerable question. This isn't a bug to patch with a better constant;
    it's a real property of lexical-overlap scoring at small scale. A single
    hardcoded threshold cannot reliably replace contextual judgment here.

    REFUSE_THRESHOLD is therefore set low (0.05) and permissive -- it only
    catches obviously-degenerate cases, and is expected to sometimes answer
    when it should refuse. This class exists to exercise pipeline plumbing
    offline (no API key needed), NOT to be a trustworthy refusal signal.
    The real refusal-quality gate is AnthropicGenerator, which uses the
    LLM's own contextual judgment via the system prompt (see
    tests/test_eval_harness.py's real-generator tests, which run only when
    API keys are available and are not part of the hard CI gate for exactly
    this reason).
    """

    REFUSE_THRESHOLD = 0.05

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
        _, source_map = build_user_prompt(query, retrieved)
        if not source_map:
            return NO_ANSWER_PHRASE, source_map

        def relevance(r: RetrievedChunk) -> float:
            return r.rerank_score if r.rerank_score is not None else r.fused_score

        best_chunk_source = max(source_map.items(), key=lambda kv: relevance(kv[1]))
        best_source_num, best_retrieved = best_chunk_source
        if relevance(best_retrieved) < self.REFUSE_THRESHOLD:
            return NO_ANSWER_PHRASE, source_map

        # Relevant enough to answer -- now pick the single sentence within the
        # winning chunk with the most query content-word overlap, so the answer
        # is at least topically on-target rather than just the chunk's first sentence.
        q_words = _content_words(query)
        best_sentence, best_score = None, -1.0
        for sentence in re.split(r"(?<=[.!?])\s+", best_retrieved.chunk.text):
            sentence = sentence.strip()
            s_words = _content_words(sentence)
            if not sentence:
                continue
            score = len(q_words & s_words)
            if score > best_score:
                best_sentence, best_score = sentence, score

        if not best_sentence:
            return NO_ANSWER_PHRASE, source_map
        return f"{best_sentence} [{best_source_num}]", source_map
