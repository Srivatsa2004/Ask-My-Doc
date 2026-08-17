"""
harness.py
----------
Core evaluation logic, shared by the CLI script (eval/run_eval.py) and the
CI-gated pytest tests (tests/test_eval_harness.py) -- one implementation,
not two copies that can silently drift apart.

Runs every question in golden_set.json through the FULL pipeline (retrieve
-> generate -> enforce), not just retrieval, and reports the metrics that
actually matter for a citation-enforcement system:

- retrieval_recall: did the right chunk get retrieved at all?
- correct_refusal_rate: for questions with NO correct answer in the docs
  (negative examples), did the system correctly decline rather than
  fabricate an answer? This is arguably the single most important number
  in the whole project -- it's the direct measurement of whether citation
  enforcement is actually preventing hallucination.
- false_refusal_rate: for questions that SHOULD be answerable, did the
  system wrongly refuse? (A citation-enforcement layer that's too
  aggressive is also a real failure mode, not just too-lenient.)
- avg_claims_dropped: how often enforcement had to intervene at all, per
  answered question -- a rough proxy for how much the generator is
  straying from the sources before enforcement catches it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.generation.pipeline import AskPipeline
from src.generation.prompt import NO_ANSWER_PHRASE

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


@dataclass
class QuestionResult:
    id: str
    question: str
    is_negative_example: bool     # True if no document actually answers this
    retrieved_chunk_ids: list[str]
    retrieval_hit: bool           # for positive examples: was an expected chunk retrieved?
    answer: str
    is_no_answer: bool            # did the system decline to answer?
    n_claims_dropped: int
    correct_behavior: bool        # the single pass/fail verdict for this question


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def positive_examples(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.is_negative_example]

    @property
    def negative_examples(self) -> list[QuestionResult]:
        return [r for r in self.results if r.is_negative_example]

    @property
    def retrieval_recall(self) -> float:
        pos = self.positive_examples
        if not pos:
            return 0.0
        return sum(r.retrieval_hit for r in pos) / len(pos)

    @property
    def correct_refusal_rate(self) -> float:
        """Of the negative examples (unanswerable questions), what fraction did
        the system correctly decline to answer, rather than fabricating a response?"""
        neg = self.negative_examples
        if not neg:
            return 1.0
        return sum(r.is_no_answer for r in neg) / len(neg)

    @property
    def false_refusal_rate(self) -> float:
        """Of the positive examples (answerable questions), what fraction did the
        system wrongly refuse to answer?"""
        pos = self.positive_examples
        if not pos:
            return 0.0
        return sum(r.is_no_answer for r in pos) / len(pos)

    @property
    def avg_claims_dropped(self) -> float:
        answered = [r for r in self.results if not r.is_no_answer]
        if not answered:
            return 0.0
        return sum(r.n_claims_dropped for r in answered) / len(answered)

    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.correct_behavior for r in self.results) / len(self.results)


def run_eval(pipeline: AskPipeline, golden_set: list[dict] | None = None) -> EvalReport:
    golden_set = golden_set if golden_set is not None else load_golden_set()
    report = EvalReport()

    for item in golden_set:
        expected_ids = item["expected_chunk_ids"]
        is_negative = not expected_ids

        result = pipeline.ask(item["question"])
        retrieved = pipeline.retriever.retrieve(item["question"], final_top_k=pipeline.retrieve_top_k)
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]

        retrieval_hit = any(cid in retrieved_ids for cid in expected_ids) if expected_ids else True

        if is_negative:
            correct = result.is_no_answer
        else:
            # correct behavior for an answerable question: retrieval found the right
            # chunk AND the system actually answered (didn't over-refuse)
            correct = retrieval_hit and not result.is_no_answer

        report.results.append(QuestionResult(
            id=item["id"], question=item["question"], is_negative_example=is_negative,
            retrieved_chunk_ids=retrieved_ids, retrieval_hit=retrieval_hit,
            answer=result.answer, is_no_answer=result.is_no_answer,
            n_claims_dropped=result.n_claims_dropped, correct_behavior=correct,
        ))

    return report
