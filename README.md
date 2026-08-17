# Ask My Docs — Enterprise Knowledge Assistant

Domain-specific document Q&A system with hybrid retrieval (BM25 + vector),
cross-encoder reranking, enforced citations, and a CI-gated evaluation
pipeline. Supports PDF, DOCX, and HTML.

## Status: Day 5 of 9 — Eval Harness + CI Gate ✅

### What's built
- **`src/ingestion/parsers.py`** — format-specific parsers for PDF (via
  `pdfplumber`, with font-size heuristics to detect headings), DOCX (via
  `python-docx`, using heading styles + table extraction), and HTML (via
  `BeautifulSoup`, stripping nav/header/footer boilerplate). All three
  normalize to a common `Section(text, heading, page)` shape.
- **`src/ingestion/chunker.py`** — structure-aware, token-budgeted chunker.
  Key correctness properties (see `tests/test_chunker.py`):
  - A chunk **never spans two PDF pages** — page citations are always exact.
  - A chunk's `section` label reflects **every** heading it actually covers
    (e.g. `"1. Overview – 3. Reimbursement Limits"`), not just the first.
  - **No content is ever silently dropped.** An early version of this
    chunker discarded small trailing sections below a token threshold —
    caught by manual testing, fixed, and now pinned by a regression test.
  - Sections longer than the chunk budget are split with a sliding window
    and configurable overlap, preserving heading/page metadata per piece.
- **`src/storage.py`** — SQLite schema for `documents` and `chunks`, with
  dedup by content hash so re-running ingestion is idempotent.
- **`scripts/ingest.py`** — CLI: `python scripts/ingest.py --input data/raw --domain enterprise_kb`
- **`tests/test_chunker.py`** — 6 tests covering the correctness properties above.

### Verified against real files
Three test documents (generated fresh, not just described) are in `data/raw/`:
an expense policy PDF (2 pages, 6 numbered sections), a remote work policy
DOCX (4 sections + a table), and an onboarding guide HTML page (with nav/
header/footer that must be stripped). All three ingest cleanly; metadata was
manually inspected and confirmed correct (see commit history / dev notes).

### Day 2 additions: Hybrid Retrieval
- **`src/retrieval/bm25_index.py`** — sparse lexical retrieval via `rank_bm25`,
  persisted to disk with pickle.
- **`src/retrieval/embeddings.py`** — `Embedder` interface with two implementations:
  `OpenAIEmbedder` (real, requires `OPENAI_API_KEY`) and `DeterministicFakeEmbedder`
  (hash-based, offline, zero network — used for dev/testing plumbing only; **not**
  semantically meaningful, so don't judge retrieval *quality* through it).
- **`src/retrieval/vector_store.py`** — Qdrant wrapper. Runs embedded/on-disk by
  default (no server needed); pass `url=`/`api_key=` to point at Qdrant Cloud instead.
- **`src/retrieval/fusion.py`** — Reciprocal Rank Fusion (k=60), the standard way to
  combine two differently-scaled ranked lists without normalizing raw scores.
- **`src/retrieval/hybrid_retriever.py`** — orchestrates build + query across both indexes.
- **`scripts/build_index.py`**, **`scripts/query.py`** — CLI tools; both support
  `--fake-embedder` for offline testing without an API key.
- **`tests/test_retrieval.py`** — 9 tests covering BM25, the vector store (real
  embedded Qdrant), RRF fusion, and the full hybrid pipeline end-to-end offline.

**Verified on real data:** ingested the 3 sample docs, built both indexes, and
queried "what is the meal reimbursement limit for domestic travel" — BM25 alone
correctly ranked the right chunk first with a clear score margin (4.25 vs 1.67).
The fused result was noisier because it was tested with the fake embedder (by
design, not semantically meaningful); this sharpens once `OPENAI_API_KEY` is set
and `OpenAIEmbedder` provides a real semantic signal.

**Known limitation, documented not silently ignored:** `rank_bm25`'s IDF formula
can hit exactly 0 for terms in tiny (~2-doc) corpora — not an issue at real
document-collection scale, but worth knowing if you're debugging with a handful
of test docs.

### Day 3 additions: Reranking + Golden Eval Set
- **`src/retrieval/reranker.py`** — `Reranker` interface with two implementations:
  `CohereReranker` (real, requires `COHERE_API_KEY`, uses `rerank-v3.5` via Cohere's
  v2 client) and `LocalOverlapReranker` (offline, Jaccard token-overlap — not a real
  cross-encoder, exists only to test reranking plumbing without a network call).
- **`src/retrieval/hybrid_retriever.py`** — updated to take an optional `reranker`.
  When set, fused candidates are rescored and the *reranker's* ordering wins over
  fusion rank — verified in tests that this correctly promotes the right chunk to
  #1 even when the fusion step was noisy.
- **`eval/golden_set.json`** — **25 questions** across all 3 sample documents, each
  with `expected_chunk_ids` and a `reference_answer`. Includes **one intentional
  negative example** (q25, asking about something no document covers) — critical
  for Day 4/5's hallucination and citation-validity checks, since a good system
  should decline to answer rather than fabricate a citation.
- **`eval/run_retrieval_eval.py`** — Recall@K eval script against the golden set.
  Built now (Day 3) rather than deferred to Day 5, so the golden set is validated
  against real retrieval behavior immediately. Day 5 wraps this in CI.
- **`tests/test_reranker.py`** — 6 tests, including golden-set structural validation
  (unique IDs, all fields present, at least one negative example).

**Verified end-to-end:** ingested → indexed → queried with reranking on, confirmed
the reranker correctly reorders fused candidates (rerank_score descending, matching
relevance — not just fusion rank). Ran the full golden set through retrieval:
24/24 scored questions hit (100% Recall@3). Note this number is inflated by the
tiny 4-chunk demo corpus (top_k=3 out of 4 total chunks retrieves almost anything);
treat it as proof the eval *plumbing* is correct, not as a real quality signal —
that becomes meaningful once Day 5 CI runs against a larger real document set.

### Day 4 additions: Citation-Constrained Generation + Enforcement
This is the core feature the project is named for. Two layers, per the
project's design decision to **strip unsupported claims rather than
regenerate the whole answer** (faster, cheaper, and the user still gets
whatever part of the answer was actually grounded):

- **`src/generation/prompt.py`** — presents retrieved chunks to the model as
  numbered sources (`[1]`, `[2]`...) rather than raw chunk_ids; the
  number→chunk mapping (`source_map`) is what enforcement verifies against.
  Instructs the model to cite every claim and to emit an exact fallback
  phrase when the sources don't answer the question.
- **`src/generation/generator.py`** — `AnthropicGenerator` (real, requires
  `ANTHROPIC_API_KEY`) plus two offline fakes: `FakeCitingGenerator` (happy
  path) and `FakeHallucinatingGenerator`, which **deliberately** cites a
  nonexistent source and states an uncited claim — built specifically to
  prove enforcement catches both failure modes, not just to pad test count.
- **`src/generation/citation_enforcer.py`** — the actual enforcement logic.
  Splits the answer into sentences, and for each one:
  1. Strips citations pointing to source numbers that don't exist (invalid).
  2. Strips sentences with no citation at all.
  3. Checks lexical overlap between the claim and its cited source's text;
     strips if the "cited" source doesn't plausibly say what's claimed
     (catches a real source number attached to a fabricated fact).
  A sentence survives only if it has at least one citation that's both
  *valid* and *supported*. If nothing survives, the answer falls back to
  the "not enough information" phrase rather than returning an empty string.
  **Documented limitation:** claim support uses lexical overlap, not a real
  NLI/entailment model — catches fabricated facts attached to a real
  citation, but won't catch subtler logical misreadings. Swappable for a
  real NLI model or LLM-as-judge later without changing the interface.
- **`src/generation/pipeline.py`** — `AskPipeline`: retrieve → generate →
  enforce → attach human-readable source references for whatever survived.
- **`scripts/ask.py`** — end-to-end CLI.
- **`tests/test_citation_enforcement.py`** — 12 tests. Caught a real bug
  during testing: the sentence splitter was separating a claim from its own
  citation marker (`"...day."` / `"[1]"` as two fragments instead of one),
  which would have caused every correctly-cited sentence to be wrongly
  stripped. Fixed and pinned by `test_correctly_cited_sentence_is_kept`.

**Verified end-to-end against real ingested data** via `scripts/ask.py --fake-all`:
- Happy path: all sources cited correctly, all survive enforcement.
- `--hallucinate` path: a fabricated `$10,000 bonus` claim with an invalid
  citation `[99]`, and an uncited "applies retroactively" claim, were both
  **caught and stripped**, while the one legitimately-cited claim survived.
  This is the core proof that citation enforcement actually works, not just
  that the prompt asks nicely for citations.

### Day 5 additions: Eval Harness + CI Gate
- **`eval/harness.py`** — shared eval logic (used by both the CLI and the CI
  tests, so there's one implementation, not two that can drift apart). Runs
  every golden-set question through the **full** pipeline (retrieve →
  generate → enforce, not just retrieval) and computes: retrieval recall,
  **correct refusal rate** (of unanswerable questions, how many did the
  system correctly decline rather than fabricate an answer — arguably the
  single most important number in the whole project), false refusal rate,
  and average claims dropped per answer.
- **`src/generation/generator.py`** — added `FakeSmartGenerator`, an offline
  stand-in that checks retrieval relevance before answering (unlike
  `FakeCitingGenerator`, which cites anything regardless of relevance and
  therefore can't test refusal behavior at all).
- **Real, documented finding from calibrating `FakeSmartGenerator`:** on
  this project's small demo corpus, there is **no single lexical-relevance
  threshold** that cleanly separates answerable from unanswerable questions
  — the one negative example scores *between* several genuinely-answerable
  questions. This isn't a bug to silently patch; it's evidence that
  hardcoded-threshold refusal is inherently unreliable at small scale,
  which is exactly why production systems rely on the LLM's own contextual
  judgment (the system prompt) instead. Documented in the class docstring
  rather than hidden.
- **Two-tier CI strategy, `tests/test_eval_harness.py`:**
  - **Tier 1 (hard gate, always runs, no API keys needed):** retrieval
    recall must be 100% on the golden set; and — the core guarantee test —
    every golden-set question is run through a generator that *deliberately
    hallucinates on every call*, asserting the fabricated content never
    survives into any final answer. A regression in either fails the build.
  - **Tier 2 (soft, auto-skips without API keys):** real end-to-end quality
    with the actual OpenAI + Cohere + Anthropic pipeline — correct refusal
    on the unanswerable question, and ≥90% overall pass rate. Runs locally
    with real keys, or in CI if configured as repo secrets; skips cleanly
    otherwise rather than failing a build that can't reach the API.
- **`.github/workflows/eval.yml`** — runs the full test suite (`pytest
  tests/ -v`) on every push/PR to `main`. Tier 1 tests are the actual gate;
  Tier 2 tests pick up repo secrets automatically if present. A separate
  offline eval report step runs for log visibility but is explicitly
  non-blocking, since it's built on the documented-noisy lexical heuristic.

**Verified:** 36/38 tests pass, 2 skip cleanly without API keys (confirmed
locally; will behave identically in CI without secrets configured).

### Try it
```bash
pip install -r requirements.txt
python scripts/ingest.py --input data/raw --domain enterprise_kb
python scripts/build_index.py --fake-embedder

# full test suite (Tier 2 auto-skips without real keys):
python -m pytest tests/ -v

# eval report:
python eval/run_eval.py --fake-all
python eval/run_eval.py --fake-all --hallucinate   # stress-test enforcement

# with real keys, Tier 2 tests run too:
export OPENAI_API_KEY=sk-...
export COHERE_API_KEY=...
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/build_index.py
python -m pytest tests/ -v
python eval/run_eval.py
```

## Roadmap
| Day | Milestone | Status |
|---|---|---|
| 1 | Ingestion + structure-aware chunking | ✅ Done |
| 2 | Hybrid retrieval: BM25 + vector (Qdrant) + RRF fusion | ✅ Done |
| 3 | Cross-encoder reranking (Cohere) + golden eval set (25 Qs) | ✅ Done |
| 4 | Citation-constrained generation + enforcement layer | ✅ Done |
| 5 | Eval harness (Recall@K, citation validity) + GitHub Actions CI gate | ✅ Done |
| 6 | FastAPI backend + Streamlit UI | ⬜ Next |
| 4 | Citation-constrained generation + enforcement layer | ⬜ |
| 5 | Eval harness (Recall@K, citation validity) + GitHub Actions CI gate | ⬜ |
| 6 | FastAPI backend + Streamlit UI | ⬜ |
| 7 | Polish, deploy (Streamlit Cloud / Render) | ⬜ |
| 8 | Hardening buffer | ⬜ |
| 9 | README, architecture diagram, demo GIF | ⬜ |

## Architecture (target, end state)
```
Docs (PDF/DOCX/HTML)
   → Ingestion & Parsing        [Day 1 ✅]
   → Structure-aware Chunking   [Day 1 ✅]
   → Embedding + BM25 indexing  [Day 2 ✅]
   → Hybrid Retrieval (RRF fusion)   [Day 2 ✅]
   → Cross-Encoder Reranking    [Day 3 ✅]
   → Context Assembly with source IDs
   → LLM Generation (citation-constrained)  [Day 4 ✅]
   → Citation Verification / Enforcement    [Day 4 ✅]
   → Response + citations → Streamlit UI    [Day 6]
```
Every change to retrieval, chunking, or prompts is checked against a fixed
golden question set in CI (Day 5) — regressions in retrieval recall or
citation validity fail the build.

## Project layout
```
ask-my-docs/
├── src/
│   ├── ingestion/
│   │   ├── parsers.py      # PDF / DOCX / HTML → Section objects
│   │   └── chunker.py      # Section objects → Chunk objects
│   ├── retrieval/
│   │   ├── embeddings.py       # OpenAIEmbedder / DeterministicFakeEmbedder
│   │   ├── bm25_index.py       # sparse lexical retrieval
│   │   ├── vector_store.py     # Qdrant wrapper (embedded or hosted)
│   │   ├── fusion.py           # Reciprocal Rank Fusion
│   │   ├── reranker.py         # CohereReranker / LocalOverlapReranker
│   │   └── hybrid_retriever.py # orchestrates build + query + rerank
│   ├── generation/
│   │   ├── prompt.py            # numbered-source prompt construction
│   │   ├── generator.py         # AnthropicGenerator / FakeCitingGenerator / FakeHallucinatingGenerator
│   │   ├── citation_enforcer.py # verify + strip unsupported claims
│   │   └── pipeline.py          # AskPipeline: retrieve -> generate -> enforce
│   └── storage.py          # SQLite: documents, chunks
├── scripts/
│   ├── ingest.py            # parse + chunk + store documents
│   ├── build_index.py       # build BM25 + vector indexes
│   ├── query.py             # ad-hoc hybrid retrieval + rerank query tool
│   └── ask.py                # end-to-end: retrieve, generate, enforce citations
├── eval/
│   ├── golden_set.json      # 25 Q&A pairs with expected citations
│   ├── harness.py            # shared eval logic (CLI + CI tests)
│   └── run_eval.py           # full-pipeline eval CLI report
├── .github/workflows/
│   └── eval.yml              # CI: runs pytest on every push/PR
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_reranker.py
│   ├── test_citation_enforcement.py
│   └── test_eval_harness.py
├── data/
│   ├── raw/                # source documents
│   └── processed/          # SQLite DB
└── requirements.txt
```
