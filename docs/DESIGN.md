# Design and Decisions: How NeoDocuMind Works (and Why)

This document explains every part of the system, the reasoning behind each
choice, the alternatives that were considered, and the trade-offs. It doubles as
a study guide: if you understand this end to end, you can defend the project in
any technical interview.

---

## 0. The big picture: what RAG is and why each stage exists

**Retrieval-Augmented Generation (RAG)** answers a question by first *finding*
relevant text from your documents, then *giving that text to an LLM* to write the
answer. It exists because LLMs have two problems:

1. **They don't know your private data** (handbooks, contracts, a specific book).
2. **They hallucinate** when asked about things outside their training.

RAG fixes both by grounding the model in retrieved evidence. The pipeline:

```
documents -> chunk -> embed -> store        (offline: "indexing")
question  -> retrieve -> rerank -> prompt -> LLM -> cited answer   (online)
```

Every stage below maps to one file in `src/documind/`. The guiding principle is
**separation of concerns with thin, swappable interfaces**, so any single piece
(vector store, LLM, embeddings) can be replaced without touching the rest.

---

## 1. Ingestion and chunking (`ingest.py`)

### What it does
Reads PDF/TXT/MD files and splits them into small, overlapping text **chunks**,
each tagged with a `citation_id` like `handbook.md#3`.

### Why chunk at all?
- **Context windows are finite and attention degrades.** You cannot (and should
  not) stuff a whole book into the prompt. You retrieve a few small, highly
  relevant pieces instead.
- **Retrieval precision.** A 300-word chunk about "PTO carryover" is a sharp
  match for a PTO question. A whole document is a blurry match for everything.

### Why this chunk size / overlap? (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`)
- Chunks too **large** -> retrieval is imprecise and you waste prompt tokens.
- Chunks too **small** -> you lose context (a sentence loses its surrounding
  meaning), and facts get split across chunks.
- ~800 characters is a balance: roughly a paragraph or two. **Overlap** (120
  chars) means a fact sitting on a chunk boundary still appears whole in at least
  one chunk, so it isn't lost.

### Why a *recursive* splitter?
`RecursiveCharacterTextSplitter` tries to split on natural boundaries first
(paragraphs `\n\n`, then lines `\n`, then sentences `. `, then words). This keeps
semantically coherent units together instead of cutting mid-sentence.

### Alternatives and trade-offs
- **Fixed-size token chunking**: simpler, but cuts across sentence boundaries.
- **Semantic chunking** (split where embedding similarity drops): higher quality
  boundaries, but slower and more complex; overkill for this scope.
- **Layout-aware parsing** (e.g. `unstructured`, tables/headings): better for
  messy PDFs, heavier dependency. A clear next step for real documents.

### Interview soundbite
> Chunking is the highest-leverage knob in RAG. I use overlapping recursive
> chunks so facts on boundaries aren't lost, and I keep chunk size tunable
> because the right value depends on the document type.

---

## 2. Embeddings (`embeddings.py`)

### What an embedding is
A model that turns text into a fixed-length vector of numbers (here 384 floats)
such that **texts with similar meaning have vectors that are close together**.
This is what makes "car" match "automobile" without sharing any letters.

### Why `BAAI/bge-small-en-v1.5`?
- **Strong quality for its size** on the MTEB retrieval benchmark.
- **Small and fast** (384 dimensions) -> cheap to store and search, runs on CPU.
- **Local and free** -> no API key, so retrieval and evaluation cost nothing and
  the whole demo is reproducible by anyone who clones it.

### Why normalize embeddings?
We set `normalize_embeddings=True`. When vectors are unit length, **L2 (Euclidean)
distance and cosine similarity rank results identically**, so FAISS's L2 search
behaves like cosine similarity, which is the standard for semantic matching.

### Alternatives and trade-offs
- **OpenAI `text-embedding-3-small/large`**: slightly better quality, but costs
  money and needs a key. The code supports swapping to it.
- **Larger local models** (`bge-large`, `e5-large`): better recall, but slower
  and more memory; not worth it for a demo, easy to switch for production.

### Interview soundbite
> Embeddings map text to a vector space where distance means semantic
> similarity. I chose a small local model so the system is free, fast, and
> reproducible, and the layer is swappable for a hosted model in production.

---

## 3. Vector store: FAISS (`vectorstore.py`)

### What it does
Stores all chunk embeddings and, given a query vector, returns the nearest ones
(this is **dense / semantic retrieval**).

### What FAISS actually is
A library for **(approximate) nearest-neighbour search** over vectors. Our
default index is exact (brute-force) which is perfect for thousands of chunks;
FAISS also offers approximate indexes (HNSW, IVF) that trade a little accuracy
for huge speed gains at millions of vectors.

### Why FAISS for this project?
- **Zero infrastructure**: it's a pip-installable library, not a server. The app
  runs anywhere (including free Streamlit Cloud) with no database to host.
- **Fast and battle-tested** (built by Meta, used widely in production).
- **Simple persistence**: save/load the index as files.

### "Why not Weaviate/Qdrant, which do vectors *and* keywords in one database?"
This is the key trade-off, and the honest answer is **operational simplicity vs.
production features**:

- **Weaviate/Qdrant pros**: built-in hybrid search (vector + BM25) in one place,
  horizontal scaling, metadata filtering, persistence, a real server you can
  share across services. In **production, this is the right choice.**
- **Their cost**: they are **servers** — you must run a container/managed
  instance, operate it, and pay for it. For a self-contained portfolio demo that
  must run for free on a single process, that's unnecessary weight.
- **My choice**: FAISS (vectors) + `rank_bm25` (keywords) gives me the *same
  hybrid capability* with **no server and no cost**, and the thin
  `vectorstore.py` interface means swapping in Qdrant/Weaviate/pgvector later is
  a localized change. The README's production section spells out that migration.

So it's a deliberate scope decision, not a knowledge gap: **right tool for a
zero-infra demo, with a clear, documented path to a managed store for scale.**

### Interview soundbite
> FAISS keeps the demo server-less and free. In production I'd move to a managed
> store like Qdrant, Weaviate, pgvector, or OpenSearch that does hybrid search,
> filtering, and scaling, which is why the vector-store layer is deliberately thin
> and swappable.

---

## 4. Sparse retrieval: BM25 (`retriever.py`)

### What it is
BM25 is a **keyword** ranking function (the modern successor to TF-IDF, the
algorithm behind classic search engines). It scores a chunk by how often the
query's words appear in it, adjusted for word rarity and document length.

### Why keep it when we already have semantic search?
Because embeddings have a real weakness: **exact tokens, rare terms, IDs,
acronyms, product codes, names.** If a user searches for "error code E-4021" or a
specific clause number, semantic similarity can drift, but BM25 nails the literal
match. BM25 and dense search fail in *different* ways, so combining them is robust.

### Interview soundbite
> BM25 is lexical; embeddings are semantic. Embeddings miss exact terms like IDs
> and acronyms; BM25 misses paraphrase. Using both covers each other's blind
> spots.

---

## 5. Hybrid retrieval and fusion (`retriever.py`)

### Why hybrid instead of pure semantic?
Pure semantic (dense-only) is the common default, but it underperforms on:
- exact-keyword queries (IDs, names, jargon),
- short queries where lexical signal matters,
- domains with terminology the embedding model wasn't trained on.

Hybrid retrieval consistently beats either method alone on mixed real-world
queries. That's why production search stacks (and managed vector DBs) all offer
hybrid.

### How the two rankings are combined: Reciprocal Rank Fusion (RRF)
We get two ranked lists (dense and sparse) and merge them with:

```
score(doc) = sum over lists of  1 / (k + rank_in_that_list)     (k = 60)
```

- It uses **rank, not raw scores**, which is crucial because BM25 scores and
  cosine similarities are on totally different scales and can't be added directly.
- A document ranked highly by **both** methods rises to the top.
- `k=60` is the standard constant from the original RRF paper; it dampens the
  influence of very low ranks.

### Alternatives and trade-offs
- **Weighted score fusion** (normalize then weight-add scores): tunable, but
  requires score normalization and is sensitive to scale; RRF is simpler and
  robust with no tuning.
- **A single hybrid-capable DB** (Weaviate/Qdrant) does this internally — same
  idea, different home (see section 3).

### Interview soundbite
> I fuse dense and sparse results with Reciprocal Rank Fusion, which combines by
> rank rather than raw score, so I don't have to normalize incomparable BM25 and
> cosine scores. It's the standard, tuning-free fusion method.

---

## 6. Reranking: cross-encoder (`retriever.py`)

### The bi-encoder vs cross-encoder distinction (this is a favourite interview topic)
- **Bi-encoder** (what embeddings are): encodes the query and each document
  **separately** into vectors, then compares with a cheap distance. Fast, because
  document vectors are **precomputed once**. But it never lets the query and
  document "look at each other," so it's less precise.
- **Cross-encoder**: feeds the **query and a document together** into a
  transformer that reads them jointly and outputs a relevance score. Much more
  accurate, because it can model fine-grained interactions ("does *this* passage
  actually answer *this* question?").

### Why use both (retrieve then rerank)?
The cross-encoder is accurate but **expensive**: it must run the model once per
(query, document) pair and **can't precompute** anything. You can't run it over
the whole corpus. So:
1. Use cheap hybrid retrieval to get the top ~12 candidates.
2. Use the cross-encoder to **re-score just those 12** and keep the best 4.

This is the standard **two-stage retrieve-then-rerank** pattern: cheap recall
first, expensive precision second.

### Why `cross-encoder/ms-marco-MiniLM-L-6-v2`?
- Trained on MS MARCO (a large query-passage relevance dataset), so it's
  purpose-built for exactly this "is this passage relevant to this query" task.
- "MiniLM-L-6" = small and fast, fine on CPU for a handful of candidates.

### Alternatives and trade-offs
- **No reranker**: faster, but noticeably noisier top results.
- **Cohere Rerank / Jina Reranker (API)**: excellent quality, but paid and
  external.
- **Larger cross-encoders** (`bge-reranker-large`): better, slower.
- **ColBERT (late interaction)**: a middle ground between bi- and cross-encoder;
  more infrastructure.
- **LLM-as-reranker**: highest quality, highest cost/latency.

### Interview soundbite
> Embeddings are a bi-encoder: fast but coarse. A cross-encoder reads the query
> and passage together so it's far more precise, but it can't be precomputed, so
> I only run it on the top candidates from the cheap retriever. Classic
> retrieve-then-rerank.

---

## 7. Generation and grounding (`pipeline.py`, `llm.py`)

### How hallucination is controlled
1. **A strict system prompt**: "answer ONLY from the provided context; if it's
   not there, say you couldn't find it."
2. **Mandatory citations**: every claim must cite a `citation_id`, which makes
   answers auditable and discourages invention.
3. **Low temperature (0.1)**: minimizes creative drift; we want faithful, not
   imaginative, answers.

### Why provider-agnostic LLM?
`llm.py` is a factory that returns OpenAI, Groq, or Ollama behind the same
interface. This avoids vendor lock-in and lets the same code run on a paid API, a
free tier, or fully local. The rest of the pipeline only knows "a chat model,"
not which one.

### Interview soundbite
> Grounding is prompt + citations + low temperature. The model is told to answer
> only from retrieved context and cite each claim, so answers are auditable and
> hallucination is minimized. The provider is swappable to avoid lock-in.

---

## 8. Evaluation (`eval/`)

### Why two tiers of metrics?
You must measure both halves of RAG separately, because they fail separately:

**Retrieval metrics** (`evaluate.py`, free/offline):
- **Hit@k**: did the correct source document appear in the top-k results? Measures
  recall.
- **MRR (Mean Reciprocal Rank)**: how *high* did the correct source rank?
  (1.0 = always first, 0.5 = usually second.) Measures ranking quality.

**Answer-quality metrics**:
- **Keyword recall**: cheap proxy — do gold keywords appear in the answer?
- **LLM-as-judge** (`judge.py`): a separate model grades each answer for
  **faithfulness** (every claim supported by context -> anti-hallucination) and
  **relevance** (does it answer the question?), scored 1-5.

### Why an LLM-as-judge?
Faithfulness can't be measured with string matching — you need something that
*understands* whether a claim is supported. An independent LLM judge is the
standard, scalable way (this is the core idea behind frameworks like RAGAS).

### Honest limitation (say this before they point it out)
The benchmark is small (12 Q/A over 3 docs), so scores are near-perfect. The
**value is the harness**, not the numbers: point it at a bigger, noisier corpus
and the metrics become discriminating. This honesty signals maturity.

### Interview soundbite
> I evaluate retrieval (Hit@k, MRR) and answers (faithfulness and relevance via
> an LLM-as-judge) separately, because retrieval and generation fail separately.
> The judge catches hallucination that keyword matching can't.

---

## 9. Serving, config, testing, CI, Docker

- **FastAPI** (`api.py`): a typed REST API (`/ask`, `/health`); the pipeline is
  loaded once at startup and reused. This is how a backend would expose RAG.
- **Streamlit** (`app/`): the demo UI with citations and document upload; cached
  with `st.cache_resource` so models load once.
- **Config** (`config.py`): all knobs in one typed, validated place
  (pydantic-settings), overridable by env vars / `.env`. No magic constants
  scattered in code.
- **Tests** (`tests/`): fast, pure-Python unit tests (chunking, fusion inputs,
  prompt building, metric math, judge parsing) that run with no model downloads,
  so CI stays fast.
- **CI** (`.github/workflows/ci.yml`): on every push, GitHub installs the project
  on Python 3.10/3.11/3.12, lints with ruff, and runs the tests. A green badge =
  automated proof the project works.
- **Docker**: containerizes the API and UI for reproducible deployment.

---

## 10. Trade-off cheat sheet

| Decision | Chosen | Main alternative | Why chosen |
| --- | --- | --- | --- |
| Chunking | Recursive, overlapping | Token/semantic chunking | Keeps facts whole, simple, tunable |
| Embeddings | Local `bge-small` | OpenAI embeddings | Free, fast, reproducible |
| Vector store | FAISS (file) | Weaviate/Qdrant/pgvector | Zero infra for demo; swappable |
| Keyword search | `rank_bm25` | DB built-in BM25 | No server needed |
| Fusion | RRF | Weighted score fusion | No score normalization/tuning |
| Reranking | Cross-encoder (MiniLM) | Cohere API / none / ColBERT | Big precision gain, runs locally |
| LLM | Provider-agnostic | Hard-coded vendor | No lock-in; free options |
| Eval | Hit@k/MRR + LLM judge | Manual eyeballing | Measurable, catches hallucination |

---

## 11. Likely interview questions (quick answers)

- **"Walk me through what happens when a user asks a question."**
  Retrieve top-12 via FAISS (semantic) and BM25 (keyword) in parallel -> fuse
  with RRF -> rerank with a cross-encoder to the top 4 -> build a grounded prompt
  with those chunks and their citation ids -> LLM writes an answer citing each
  claim.

- **"Why hybrid and not just embeddings?"** Embeddings miss exact tokens (IDs,
  acronyms, names); BM25 catches them. They fail differently, so combining is
  more robust.

- **"Why a cross-encoder if you already retrieved?"** Retrieval (bi-encoder) is
  fast but coarse; the cross-encoder reads query+passage jointly for precision.
  It's too expensive to run on everything, so only on the top candidates.

- **"How do you stop hallucination?"** Grounding prompt (answer only from
  context), mandatory citations, low temperature — and I *measure* it with an
  LLM-judge faithfulness score.

- **"How would you scale this to millions of documents?"** Move to a managed
  hybrid vector store (Qdrant/Weaviate/pgvector/OpenSearch) with approximate
  search; run ingestion as an async job; serve a stateless API with autoscaling;
  cache embeddings; keep the eval suite in CI as a quality gate.

- **"What would you improve next?"** Streaming responses, semantic/layout-aware
  chunking, metadata filtering, multi-turn query rewriting, and a larger labelled
  eval set.
```
