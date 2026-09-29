# INTERVIEW PREPARATION – QA Knowledge Bot

## 0. Say-it-honestly checklist (read this first)

| ✅ Actually implemented | ❌ NOT implemented (never claim these) |
|---|---|
| PDF/DOCX/TXT ingestion, word-based chunking with overlap | Re-ranking, hybrid/BM25 search, query rewriting |
| MiniLM embeddings, Qdrant storage with full payload | Conversation memory (each question is independent) |
| Top-K + similarity threshold, refusal *before* calling the LLM | OCR for scanned PDFs |
| Grounded system prompt, `[Source n]` citations, source viewer | Verified citations / faithfulness checking |
| Duplicate detection (content hash), delete, clear | Authentication, multi-user isolation |
| Unit tests with mocks (40 tests) | Retrieval-quality benchmark (recall@k etc.) |
| Local (embedded) Qdrant and Docker Qdrant option | Load/scale testing; streaming responses |

Things to be candid about if asked: I tested the pipeline with a fake embedding model + real in-memory Qdrant + fake LLM; the real MiniLM model and real Groq calls are exercised only when *you* run the app with internet and a key. The similarity threshold (0.30) is a starting heuristic, not a calibrated value. The Docker files are untested by the author.

---

## 1. Beginner explanations (the 26 topics)

1. **What problem does it solve?** LLMs don't know your private documents and may invent answers. This app lets you ask questions about *your* files and get answers with visible sources.
2. **What is RAG?** Retrieve relevant passages from a knowledge base, then Generate an answer with an LLM that only sees those passages.
3. **Why not just send the question to Groq?** The model doesn't have your documents, can't cite them, and will guess. Pasting whole documents into the prompt doesn't scale (context limits, cost, and relevant details get lost).
4. **What is an embedding?** A list of numbers (384 for MiniLM) representing the *meaning* of a text. Texts with similar meaning get vectors that point in similar directions.
5. **Why Sentence Transformers?** It's a mature library giving pre-trained models that turn sentences into vectors, runs locally on CPU, free, no API calls for embeddings, and the models are trained specifically so sentence similarity works.
6. **What does all-MiniLM-L6-v2 do?** A small (6-layer) transformer, fine-tuned on ~1B sentence pairs so similar sentences map close together. Outputs 384-dim vectors; max input 256 word-pieces (longer text is truncated). Fast, good general-purpose English baseline; not state of the art.
7. **What is vector similarity?** Measuring how close two vectors are; close = similar meaning.
8. **What is cosine similarity?** The cosine of the angle between two vectors: 1 = same direction, 0 = unrelated, negative = opposite. It ignores vector length. With normalised vectors it equals the dot product.
9. **What is Qdrant?** An open-source vector database: stores vectors + JSON payloads and does fast nearest-neighbour search with filtering.
10. **Why a vector database?** Storing vectors persistently, searching them efficiently (approximate nearest neighbour indexes like HNSW in server deployments), filtering by metadata, and keeping text/metadata next to the vector.
11. **What's stored in Qdrant?** Per chunk: a point ID (UUID derived from the chunk ID), the 384-float vector, and payload `{document_id, filename, file_type, page, chunk_id, chunk_index, uploaded_at, text}`.
12. **Upload flow?** Validate → SHA-256 hash (duplicate check) → save file → extract text per page → chunk → embed all chunks in batches → upsert to Qdrant.
13. **Question flow?** Embed question → Qdrant top-K search → drop chunks below threshold → if none, refuse; else build context → Groq → answer + sources.
14. **How is the question converted?** Same MiniLM model, `embed_query()`, normalised to unit length.
15. **How does retrieval work?** Qdrant compares the query vector with stored vectors using cosine distance and returns the K highest-scoring points with their payloads.
16. **Top-K?** How many best-matching chunks to keep. Too small → may miss the answer; too large → noise, cost, and distracting context. Default 5, slider 1–10.
17. **Chunking?** Splitting documents into passages small enough to embed precisely and to fit in a prompt.
18. **Why overlap?** So information that crosses a chunk boundary still appears intact in at least one chunk.
19. **How is attribution implemented?** Metadata is attached at chunking time, stored in the Qdrant payload, returned with each hit, numbered `[Source n]` in the prompt, and displayed under the answer. The LLM is asked to cite labels; the UI marks which were cited.
20. **How are hallucinations reduced?** Threshold-gated refusal (LLM not called with irrelevant context), strict prompt, low temperature, visible sources. Reduced, **not eliminated**.
21. **If nothing relevant?** Fixed message: "I couldn't find sufficient information in the uploaded documents to answer this question." No LLM call.
22. **Why is metadata important?** It's the only thing that lets you say *where* an answer came from, enables filtering/deleting per document, and supports debugging retrieval.
23. **Why Groq?** Very fast inference for open-weight models via a simple API and a free tier – good for interactive demos. (The design is LLM-agnostic: only `groq_client.py` would change.)
24. **Retrieval vs generation?** Retrieval finds relevant text (embedding math, no language generation). Generation writes the natural-language answer from that text. Most RAG failures are retrieval failures.
25. **Limitations?** See the table in section 0 and README §11.
26. **Production improvements?** Hybrid search, re-ranking, evaluation set, auth, async ingestion, server-mode Qdrant with replication, monitoring, caching, streaming, structure-aware chunking, OCR, guardrails for prompt injection.

---

## 2. Explanations to say out loud

### C. 60-second version
"QA Knowledge Bot is a document question-answering app built on Retrieval-Augmented Generation. You upload PDFs, Word or text files; I extract the text, split it into overlapping chunks, embed each chunk with Sentence Transformers' MiniLM model, and store the vectors with metadata in Qdrant. When you ask a question, I embed it with the same model, search Qdrant for the most similar chunks, and send only those chunks to an LLM on Groq with a prompt that says 'answer only from this context'. The UI shows the answer plus the source file, page and chunk text. If nothing is similar enough, the app refuses without even calling the LLM, which reduces hallucination. It's modular, has unit tests with mocks, and runs locally."

### D. 2-minute version
Add to the above: "The code is layered: loaders → chunker → embedding service → Qdrant store → retriever → context builder → Groq client, orchestrated by a QAService, with Streamlit as a thin UI. Chunks are 180 words with 30 overlap because MiniLM truncates at 256 tokens; I chose per-page chunking for PDFs so every chunk has an exact page number. Document IDs are content hashes so duplicates are detected, and chunk IDs are deterministic so re-indexing is idempotent. The threshold gate, the grounded prompt with untrusted-context instruction, and low temperature are my hallucination mitigations – they reduce but don't eliminate the problem. Errors map to friendly messages while technical details go to logs. Known limitations: single-turn, no re-ranking, no OCR, uncalibrated threshold."

### E. 5-minute deep technical version (outline to speak through)
1. **Ingestion** – content-hash ID → loader (`pypdf` per page / `python-docx` paragraphs+tables / UTF-8 with latin-1 fallback) → `Chunker` (word windows, step = size − overlap, tiny tail merged, deterministic IDs `<doc>-<index>`).
2. **Embedding** – single lazily-loaded `SentenceTransformer`, batch encode, `normalize_embeddings=True` so cosine = dot product; dimension read from the model, collection created with `Distance.COSINE`; startup check rejects a dimension mismatch.
3. **Storage** – `PointStruct(id=uuid5(chunk_id), vector, payload)`; upsert in batches of 64; UUIDv5 means the same chunk always maps to the same point (idempotent).
4. **Retrieval** – `query_points` with `limit=K`, `score_threshold`; results become `RetrievedChunk` with metadata; retriever has no LLM dependency so it's testable alone.
5. **Context** – dedupe (exact/contained), size cap (6000 chars), `[Source n]` labels with file and page (or chunk ID).
6. **Generation** – system prompt with 6 rules; context in `<context>` tags; temperature 0.1; error translation for auth/rate-limit/model-not-found/connection.
7. **Attribution** – all context chunks shown; regex on `Source\s*(\d+)` marks cited ones.
8. **Testing** – fake embedding model injected via constructor, real Qdrant engine in memory, fake LLM; tests for metadata surviving PDF page → Qdrant → retrieval.
9. **Trade-offs & limits** – dense-only retrieval, heuristics, single-turn.

### F. Architecture explanation
Ingestion path (offline, once per file): Loader → Chunker → Embedder → Qdrant. Query path (online, per question): Embedder → Qdrant search → Context Builder → Groq → Answer + Sources. The two paths share the embedding model and the metadata schema (`schemas.py`); that shared contract is what makes attribution possible.

### G. End-to-end request flow (question)
`chat_input` → `QAService.ask` → empty check → `store.count()==0`? (tell user to upload) → `Retriever.retrieve` → `EmbeddingService.embed_query` → `QdrantStore.search` (top-K, threshold) → empty? refuse : `ContextBuilder.build` → `GroqClient.generate` → parse `[Source n]` → `Answer(answer, sources)` → UI renders markdown + expanders.

### H. Technologies and why
| Tech | Why chosen | Alternative |
|---|---|---|
| Python 3.11 | Ecosystem for ML/RAG | – |
| Sentence Transformers | Local, free, simple API | OpenAI/Cohere embeddings (paid, network) |
| MiniLM-L6-v2 | Small, fast, 384-dim, decent quality | bge/e5/mpnet (better, bigger) |
| Qdrant | Real vector DB, payloads, filters, embedded mode | FAISS (library, no payloads), pgvector, Pinecone |
| Groq | Very fast inference, easy SDK | OpenAI, Anthropic, local Ollama |
| Streamlit | Fastest way to a clean chat UI | FastAPI + React |
| Pydantic | Typed models shared across layers | dataclasses |
| pypdf / python-docx | Pure-Python parsing | PyMuPDF, unstructured |

---

## 3. 30+ likely interview questions with truthful answers

**Q1. Explain the project in one sentence.** A RAG app that answers questions from uploaded documents using MiniLM embeddings, Qdrant search and a Groq LLM, showing sources for each answer.

**Q2. Why RAG rather than fine-tuning?** Fine-tuning teaches style/skills, not reliably facts, is costly, and can't cite sources or update instantly. RAG updates by adding a file and gives provenance.

**Q3. How did you choose chunk size?** I measure in words, default 180/30, because MiniLM truncates beyond 256 word-pieces (~190 words); bigger chunks would be partially ignored by the embedding. It's configurable and I haven't run a formal tuning experiment.

**Q4. Why overlap?** Boundary coverage: a fact split across two chunks would otherwise be half in each. Cost: duplicate storage and possible duplicate retrieval (which the context builder dedupes only when text is identical/contained).

**Q5. Why chunk per page for PDFs?** So each chunk has exactly one page number for citations. Downside: text that spans a page break is split.

**Q6. Why must query and document embeddings use the same model?** Each model defines its own vector space; distances between vectors from different models are meaningless, and dimensions typically differ.

**Q7. Why cosine similarity?** It compares direction (meaning) not magnitude, and sentence-embedding models are trained with it. With normalised vectors it's equivalent to dot product.

**Q8. Why normalise embeddings?** Keeps scores in a consistent range (−1..1), makes cosine = dot product, and avoids length effects.

**Q9. What is the similarity threshold and how did you choose 0.30?** A minimum cosine score for a chunk to count as relevant. 0.30 is a conservative starting point for MiniLM on English text; I did not calibrate it on a labelled set. It's a slider so users can tune it, and a real project should choose it from evaluation data.

**Q10. What if a question is related to the docs but the answer is missing?** The chunk may pass the threshold, so the LLM is the second line of defence: the prompt requires "The answer is not available in the uploaded documents." That depends on the LLM obeying the prompt – not guaranteed.

**Q11. Does your system eliminate hallucinations?** No. It reduces them via gating, a strict prompt, low temperature and visible sources. The LLM can still misinterpret context, and I don't verify faithfulness automatically.

**Q12. How is source attribution done, and is it reliable?** Metadata flows from ingestion to the Qdrant payload to the UI, so *retrieved* sources are exact. Whether the LLM *used* a source is inferred from `[Source n]` markers it writes – not verified, so I label sources "cited" vs "retrieved".

**Q13. What is stored in Qdrant?** Vector (384 floats), plus payload with document_id, filename, file_type, page, chunk_id, chunk_index, uploaded_at, and the chunk text.

**Q14. Why store text in the payload instead of a separate database?** One lookup returns everything needed; simpler for a small system. At larger scale you might store text in a document store and keep only IDs in the vector DB.

**Q15. How do you handle duplicate uploads?** `document_id` is the first 16 hex chars of SHA-256 of the file bytes; if that ID already exists in Qdrant, the file is skipped. Same content under a different name is still a duplicate; a slightly edited file is a new document.

**Q16. Why UUIDv5 for point IDs?** Qdrant requires UUID/integer IDs. Deriving a UUID from the chunk ID makes upserts idempotent.

**Q17. Qdrant local mode vs server?** Local mode embeds the engine in the Python process, persists to a folder, needs no setup, but only one process can open the folder and it isn't meant for large scale. Server mode (Docker) supports concurrent clients, HNSW indexing and production features. Switch by setting `QDRANT_URL`.

**Q18. How does the vector search scale?** For a server deployment Qdrant uses HNSW approximate nearest-neighbour search, sub-linear in collection size. I haven't load-tested this project; in embedded mode small collections are searched by brute force, which is fine at demo scale.

**Q19. Why is embedding at ingestion time?** Embedding is the expensive step; doing it once per chunk means a query only needs one small embedding plus a fast index lookup.

**Q20. Why load the model once?** Loading takes seconds and hundreds of MB of memory; Streamlit's `cache_resource` plus lazy loading in `EmbeddingService` guarantee a single instance per process.

**Q21. How do you prevent the LLM using outside knowledge?** Prompt rules + refusing before the call when nothing is relevant + low temperature. Prompts are not a hard guarantee.

**Q22. What about prompt injection in documents?** The prompt tells the model to treat context as data and ignore instructions inside it, and context is delimited with tags. That is a mitigation, not a robust defence; production would add filtering/monitoring.

**Q23. Why Groq?** Fast, simple API, generous free tier, open-weight models. The LLM is behind `GroqClient.generate(question, context)`, so swapping providers touches one file.

**Q24. How do you handle a deprecated Groq model?** Model name is `GROQ_MODEL`; a `NotFoundError` is translated to a message telling the user to set a current model. Default is `llama-3.3-70b-versatile`, which Groq listed as a production model when I built this.

**Q25. What error handling exists?** Custom exceptions with a `user_message` (shown) and `detail` (logged): missing key, bad/empty/unsupported/too-large files, encrypted PDFs, Qdrant connection/lock, embedding failures, Groq auth/rate/connection/model errors, no results. Ingestion returns per-file statuses so one bad file doesn't stop others.

**Q26. How did you test it without external services?** Dependency injection: `EmbeddingService(model=fake)`, Qdrant in `:memory:` mode (real engine), `FakeLLM`. Tests verify chunking, metadata survival, dimensions, retrieval order, threshold refusal (LLM call count = 0), duplicates, malformed files, and error mapping.

**Q27. What are the weaknesses of dense retrieval?** Misses exact identifiers/rare keywords, is sensitive to chunking, and may rank topically-similar-but-irrelevant text high. Fixes: hybrid search, re-ranking, better chunking.

**Q28. How would you evaluate the system?** Build a labelled Q&A set from your docs; measure retrieval recall@k/MRR, then answer correctness and faithfulness (LLM-as-judge or human review); tune chunk size, K and threshold on that set. *Not implemented here.*

**Q29. How would you support multi-turn chat?** Keep history, rewrite follow-up questions into standalone queries with an LLM before retrieval, and include limited history in the prompt. Not implemented.

**Q30. How would you make it multi-user?** Auth, per-user (or per-tenant) collections or a `user_id` payload filter, per-user upload storage, quotas. Not implemented.

**Q31. What would you change for production?** Qdrant server with persistence/replication, async ingestion worker, FastAPI backend, hybrid search + re-ranker, evaluation pipeline, observability (latency, retrieval scores), rate limiting, secrets manager, CI with tests, prompt-injection guardrails.

**Q32. What was the hardest design decision?** Chunking: page-level chunks give exact citations but split cross-page passages, and the MiniLM 256-token limit forced smaller chunks than the common 500–800 default.

**Q33. Why Streamlit not FastAPI?** For a single-user demo, Streamlit avoids a second service. Business logic sits in `QAService`, so a FastAPI layer could be added without rewriting it.

---

## 4. Follow-up and difficult questions (with short answers)

**I. Possible follow-ups**
- *"How would you pick K?"* Measure recall@k on labelled questions; balance recall vs prompt noise/cost.
- *"What if two chunks contradict?"* Currently the LLM decides; better: show both with sources, prompt to flag conflicts.
- *"What if the answer spans multiple chunks?"* Top-K provides several chunks; failures happen when pieces aren't all retrieved – motivates larger K, re-ranking or multi-hop retrieval.
- *"How do you update a document?"* Delete the old one (sidebar 🗑) and upload the new version.

**J. Difficult questions**
- *"Your threshold is arbitrary."* True – it's a heuristic default; I'd calibrate it with labelled positives/negatives and maybe use per-query relative scores.
- *"How do you know citations are correct?"* I don't verify them automatically. I show retrieved chunks so a human can check, and mark LLM-cited ones separately.
- *"Isn't showing all retrieved chunks as sources misleading?"* That's why they're labelled cited/retrieved.
- *"Why not just use a long-context model?"* Cost/latency grow with context, relevant details get lost in long contexts, corpora exceed any window, and you lose per-passage provenance.
- *"What if the PDF text extraction is garbled?"* Then embeddings are poor; I'd try another parser (e.g. PyMuPDF), layout-aware extraction or OCR.
- *"What happens with 10,000 documents?"* Use server Qdrant; ingestion should be async; `list_documents` (scroll over all points) would need a separate metadata table; re-ranking becomes important.

---

## 5. Technology-specific questions

### K. Sentence Transformers
- *What is it?* A library wrapping transformer encoders fine-tuned (bi-encoder, contrastive loss) to produce sentence embeddings.
- *Bi-encoder vs cross-encoder?* Bi-encoder embeds texts independently (fast, indexable); cross-encoder scores pairs jointly (more accurate, slow) – used for re-ranking. I use only a bi-encoder.
- *Why 384 dims?* It's the model's output size; larger models give 768/1024.
- *Max sequence length?* 256 for MiniLM-L6-v2; longer input truncated silently.
- *Does it need GPU?* No, CPU is fine at this scale.
- *Multilingual?* This model is English-focused; use a multilingual model for other languages.

### L. Qdrant
- *Distance metrics?* Cosine, dot, Euclid, Manhattan; I use cosine.
- *What is a payload?* JSON metadata attached to a point, filterable.
- *What is HNSW?* A graph-based ANN index giving fast approximate search.
- *Collection vs point?* Collection = table-like container with fixed vector size/metric; point = vector + payload + ID.
- *Why check dimension on startup?* Changing the embedding model would otherwise cause insertion errors or, worse, nonsense.
- *Filtering by metadata?* Used for per-document existence check and delete via `FieldCondition`.

### M. RAG
- *Typical failure modes?* Bad chunking, wrong retrieval, too much/too little context, LLM ignoring context, stale index.
- *Naive vs advanced RAG?* Mine is naive RAG (embed → top-K → generate). Advanced adds query rewriting, hybrid search, re-ranking, evaluation.
- *Why threshold and top-K?* Top-K bounds cost/noise; threshold lets the system say "I don't know".

### N. Embeddings
- *Do embeddings understand negation/numbers well?* Weakly – "20 days" and "10 days" are close in vector space, which is why the LLM reads the actual text.
- *Can embeddings leak information?* They can partially reveal source text; treat the vector DB as sensitive.
- *Why normalise?* See Q8.

### O. Groq
- *What is Groq?* An inference provider running open-weight models on custom LPU hardware with an OpenAI-style chat API.
- *Which model and why?* `llama-3.3-70b-versatile` by default – strong instruction-following; `llama-3.1-8b-instant` is a faster/cheaper option. Configurable.
- *Rate limits?* Free tier has limits; I map rate-limit errors to a friendly message but don't implement retries/backoff.
- *Temperature?* 0.1 for deterministic, faithful answers.

### P. Hallucinations
- *Definition?* Fluent output not supported by the provided context or reality.
- *Two kinds here?* Ungrounded (invented) vs unfaithful (misreads context).
- *What mitigations exist beyond mine?* Faithfulness checks/NLI, citation verification, answer-span extraction, self-consistency, human review.

### Q. Scalability and production
- *Bottlenecks?* Embedding throughput at ingestion (batch/GPU/queue), LLM latency and rate limits, single-process embedded Qdrant.
- *Horizontal scaling?* Stateless app replicas + Qdrant cluster; ingestion workers via a queue; cache frequent queries/embeddings.
- *Security?* Secrets in env/secret manager, auth, per-tenant isolation, upload validation and size limits (some present), PII policy, audit logs.
- *Cost?* Embeddings are free (local); cost is Groq tokens ≈ K chunks × chunk size per question.

---

## 6. Demo tips for the interview
1. Show the sample handbook, index it, ask "How many days of annual leave…?" and open the source expander.
2. Ask "What is the dental insurance coverage?" and "What is the capital of France?" – explain the two defence layers.
3. Move the threshold slider to show its effect.
4. Open `qa_service.py` and walk the ingestion and `ask` methods; open `tests/test_qa.py` to show mocking.
5. Volunteer the limitations before being asked – it reads as maturity.
