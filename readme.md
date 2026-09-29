# 🇻🇳 Vietnam Legal RAG

Vietnam Legal RAG is a Vietnamese legal question-answering system built around a local Retrieval-Augmented Generation (RAG) pipeline.

The project is designed to retrieve relevant Vietnamese legal documents, rerank the retrieved passages, rebuild their legal hierarchy, filter special-case noise, and finally use a local Qwen3-4B model to generate a structured answer with citations.

The current architecture is intentionally modular so that the retrieval system, context construction, answer generation, FastAPI backend, and Next.js frontend can evolve independently.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Environment Setup](#environment-setup)
7. [Python Dependencies](#python-dependencies)
8. [Local Qwen3-4B Setup](#local-qwen3-4b-setup)
9. [Qdrant Setup](#qdrant-setup)
10. [Dataset and Storage](#dataset-and-storage)
11. [RAG Pipeline](#rag-pipeline)
12. [Retrieval Pipeline Details](#retrieval-pipeline-details)
13. [Context Builder](#context-builder)
14. [Context Selector](#context-selector)
15. [Answer Generation](#answer-generation)
16. [FastAPI Backend](#fastapi-backend)
17. [Next.js Frontend](#nextjs-frontend)
18. [Running the Complete System](#running-the-complete-system)
19. [Running Individual Tests](#running-individual-tests)
20. [Testing the Full RAG Pipeline](#testing-the-full-rag-pipeline)
21. [Example Questions](#example-questions)
22. [Configuration](#configuration)
23. [Development Workflow](#development-workflow)
24. [Troubleshooting](#troubleshooting)
25. [Design Decisions](#design-decisions)
26. [Current Limitations](#current-limitations)
27. [Roadmap](#roadmap)

---

# Project Overview

The goal of this project is to build an offline/local Vietnamese legal assistant that can answer natural-language legal questions using a Vietnamese legal document corpus.

The system does **not** rely on a single model to "know" the law. Instead, it follows a retrieval-first architecture:

```text
User question
      │
      ▼
Query Normalizer
(Qwen3-4B)
      │
      ▼
Retriever
      │
      ├── multilingual-e5-base
      └── Qdrant
      │
      ▼
Candidate chunks
      │
      ▼
Vietnamese Reranker
      │
      ▼
Context Builder
      │
      ▼
Context Selector / Filter
      │
      ▼
Selected legal context
      │
      ▼
Answer Generator
(Qwen3-4B)
      │
      ▼
Structured LegalAnswer
      │
      ├── answer
      ├── citations
      ├── insufficient_evidence
      └── note
```

The architecture is designed to reduce hallucination risk by requiring the answer generator to rely on retrieved legal evidence.

---

# Features

## Retrieval

- Vietnamese legal document retrieval using `intfloat/multilingual-e5-base`.
- Vector search through Qdrant.
- Original, normalized, and enriched query signals can be used during retrieval.
- Exact chunk IDs are deterministic and preserved throughout the pipeline.

## Reranking

- Vietnamese reranking with `AITeamVN/Vietnamese_Reranker`.
- Batch inference on CUDA.
- Reranker scores are treated as relative relevance signals.
- Exact duplicate chunks can be removed without treating equal scores as duplicates.

## Context Construction

- Groups retrieved chunks by legal document.
- Groups content by article.
- Preserves article → clause → point hierarchy.
- Sorts legal positions numerically where possible.
- Keeps legal metadata such as title, legal type, issuing authority, and issuance date.

## Context Selection

- Reduces excessive retrieval noise before generation.
- Uses reranker score as the primary relevance signal.
- Uses lexical overlap as a supporting signal.
- Uses `issuance_date` only as a weak recency signal.
- Includes special-case awareness for conditions such as:
  - domestic workers;
  - special occupations/jobs;
  - specific fixed-term contract conditions.
- Does not assume that the newest issuance date automatically means "legally effective".

## Answer Generation

- Local Qwen3-4B generation.
- Structured JSON output.
- Pydantic validation.
- Legal citations include document and chunk references.
- Explicit `insufficient_evidence` state for cases where the context is not sufficient.

## API and UI

- FastAPI backend.
- Next.js / React frontend.
- Chat-style UI.
- Citation cards.
- RAG debug information.
- Browser-based interaction without directly exposing Qdrant or model internals to the frontend.

---

# Architecture

## High-level architecture

```text
┌───────────────────────────────────────────────────────────┐
│                       Next.js / React                     │
│                                                           │
│   Chat UI • Answer • Citations • Debug Information       │
└──────────────────────────────┬────────────────────────────┘
                               │ HTTP
                               ▼
┌───────────────────────────────────────────────────────────┐
│                         FastAPI                           │
│                                                           │
│                    POST /api/chat                         │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│                    LegalRAGPipeline                       │
│                                                           │
│  Query Normalizer                                         │
│        ↓                                                  │
│  Retriever ────────────────► Qdrant                       │
│        ↓                                                  │
│  Reranker                                                 │
│        ↓                                                  │
│  Context Builder                                           │
│        ↓                                                  │
│  Context Selector                                          │
│        ↓                                                  │
│  Answer Generator ────────► Local Qwen3-4B               │
└───────────────────────────────────────────────────────────┘

                 Supporting storage
                 ┌─────────────────┐
                 │ SQLite          │
                 │ ChunkStore      │
                 └─────────────────┘
```

---

# Project Structure

The exact set of helper scripts may grow over time, but the main architecture is:

```text
Vietnam-Legal-Chatbot/
│
├── src/
│   └── vn_legal_rag/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── llm/
│       │   └── llama_cpp.py
│       │
│       ├── embedding/
│       │   └── model.py
│       │
│       ├── query/
│       │   └── normalizer.py
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── qdrant_store.py
│       │   ├── index_builder.py
│       │   ├── chunk_store.py
│       │   ├── retriever.py
│       │   └── reranker.py
│       │
│       ├── context/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── builder.py
│       │   └── selector.py
│       │
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── prompt.py
│       │   └── generator.py
│       │
│       └── pipeline.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── dependencies.py
│
├── scripts/
│   └── ...
│
├── tests/
│   ├── ...
│   ├── test_context_builder.py
│   ├── test_context_selector.py
│   ├── test_answer_schema.py
│   ├── test_answer_generator.py
│   ├── test_context_integration.py
│   └── ...
│
├── models/
│   └── Qwen3-4B-Q4_K_M.gguf
│
├── data/
│   └── ...
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── ...
│
├── pyproject.toml
└── README.md
```

> The repository may contain additional ingestion/data-preparation scripts and generated data. Do not move or rename existing project directories unless the project architecture is intentionally being redesigned.

---

# Requirements

## Recommended hardware

The project is designed around a local CUDA workflow.

The development machine used during implementation includes:

- NVIDIA GPU with CUDA support.
- RTX 3070 8 GB was used during development.
- The pipeline is also intended to remain usable on an RTX 4060 Laptop GPU 8 GB class device, with appropriate batch/context settings.

RAM usage depends heavily on the data preparation and embedding stages. Qdrant is used as the vector database so that the application does not need to keep the entire vector index in Python RAM.

## Software

Recommended environment:

- Windows 10/11 or Linux
- Python 3.11+
- Node.js / npm for the Next.js frontend
- Docker Desktop for Qdrant
- NVIDIA driver with CUDA support
- Git
- PowerShell on Windows

---

# Environment Setup

## 1. Clone the repository

```powershell
git clone <your-repository-url>
cd Vietnam-Legal-Chatbot
```

If the project already exists locally:

```powershell
cd D:\Code\Vietnam-Legal-Chatbot
```

---

## 2. Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
pip --version
```

---

# Python Dependencies

The main project dependencies are declared in `pyproject.toml`.

Install the project in editable mode:

```powershell
pip install -e .
```

This installs the Python package and its regular dependencies.

The project uses:

- `transformers`
- `huggingface_hub`
- `sentence-transformers`
- `torch`
- `datasets`
- `pyarrow`
- `numpy`
- `pandas`
- `qdrant-client`
- `sentencepiece`
- `tiktoken`
- `fastapi`
- `uvicorn`
- `pydantic`
- `python-dotenv`
- `tqdm`
- `loguru`

Development tools such as pytest, Ruff, and Black may be installed through the project's development extras when configured.

---

# llama-cpp-python CUDA Installation

`llama-cpp-python` is intentionally installed separately because the project uses a CUDA-specific wheel.

Install the CUDA 12.5 build:

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

Verify the installation:

```powershell
python -c "from llama_cpp import Llama; print('llama_cpp OK')"
```

A successful CUDA initialization should show the NVIDIA GPU being detected.

Example:

```text
ggml_cuda_init: found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 3070
```

If `llama_cpp` imports correctly but CUDA is not detected, do not immediately reinstall the entire project. First verify the installed `llama-cpp-python` wheel, NVIDIA driver, and CUDA-compatible environment.

---

# Local Qwen3-4B Setup

The answer generator and query normalizer use a local Qwen3-4B GGUF model.

Expected file:

```text
models/
└── Qwen3-4B-Q4_K_M.gguf
```

The configured model path should be controlled through `config.py` rather than hard-coded in application modules.

## Download with Hugging Face CLI

Install the Hugging Face CLI if necessary:

```powershell
pip install -U huggingface_hub
```

Download the model:

```powershell
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models
```

After downloading, verify:

```powershell
dir models
```

You should see:

```text
Qwen3-4B-Q4_K_M.gguf
```

The model is loaded locally by `LocalLLM` from `src/vn_legal_rag/llm/llama_cpp.py`.

---

# Qdrant Setup

Qdrant runs through Docker and stores the vector index separately from the Python process.

## Start Qdrant

Example Docker command:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Keep this container running while using the RAG application.

Verify that Docker sees the container:

```powershell
docker ps
```

Qdrant should be available at:

```text
http://localhost:6333
```

The web dashboard can be opened at:

```text
http://localhost:6333/dashboard
```

## Important workflow rule

Starting/stopping the Python application is independent from the Qdrant container.

Typical workflow:

```text
Start Docker / Qdrant
        ↓
Activate Python virtual environment
        ↓
Run the required Python script or FastAPI
        ↓
Stop Python application when finished
        ↓
Leave Qdrant running for later sessions
```

If Qdrant data is stored in a Docker volume, the collection persists across container restarts.

---

# Dataset and Storage

The project uses a large Vietnamese legal document corpus.

The pipeline separates vector storage from text storage:

```text
                        chunks.jsonl
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
       Contextual embedding       SQLite ChunkStore
                │                       │
                ▼                       │
              Qdrant                   │
                │                       │
                └───────────┬───────────┘
                            ▼
                         Retrieval
```

## Why two stores?

### Qdrant

Stores:

- vector embeddings;
- retrieval metadata;
- deterministic point IDs;
- document and hierarchy metadata needed during retrieval.

### SQLite / ChunkStore

Stores:

- full chunk text;
- complete chunk metadata;
- source information needed after retrieval.

This design avoids putting the full text and all application metadata into the vector database.

---

# Chunk Model

A legal chunk contains fields such as:

```text
chunk_id
document_id

article
article_no

clause
clause_no

point
point_no

chunk_index
sub_chunk_index

start_char
end_char

title
legal_type
legal_sectors
issuing_authority
issuance_date

url
signers

text
```

The deterministic `chunk_id` is important because it is used to connect:

```text
Qdrant point
      ↓
RetrievedChunk
      ↓
ContextChunk
      ↓
Citation
      ↓
SQLite source text
```

---

# RAG Pipeline

The full RAG pipeline consists of the following stages.

## 1. Query Normalizer

The user's natural-language question is normalized by a local Qwen3-4B model.

The normalizer preserves:

- what the user is asking;
- legal subject/object;
- conditions;
- dates and years;
- quantities;
- exceptions;
- temporal constraints.
---

# Retrieval Pipeline Details

## 2. Embedding

The project uses:

```text
intfloat/multilingual-e5-base
```

Embedding dimensionality:

```text
768
```

For E5 models:

- queries use a `query:` prefix;
- passages use a `passage:` prefix.

The embedding layer handles those prefixes internally.

---

## 3. Contextual Embedding

A chunk is embedded together with its legal hierarchy.

Instead of embedding only:

```text
17. Bỏ trốn sau khi gây tai nạn để trốn tránh trách nhiệm.
```

the system can embed contextual information such as:

```text
Điều 8. Các hành vi bị nghiêm cấm
Khoản 17
17. Bỏ trốn sau khi gây tai nạn để trốn tránh trách nhiệm.
```

This helps semantic retrieval understand the legal context surrounding a short clause or point.

---

## 4. Qdrant Retrieval

The retriever searches Qdrant using one or more query representations.

The current retrieval design can preserve:

```text
Original query
Normalized query
Enriched normalized query
```

The enrichment may include:

```text
question_intent
keywords
constraints
temporal_constraints
```

The goal is to avoid losing information contained in the original user phrasing during query rewriting.

---

## 5. Candidate Retrieval

Example:

```text
Top 30 candidates
```

The exact number should remain configurable.

The retriever then restores full chunk records from SQLite.

---

# Reranker

## 6. Vietnamese Reranker

The project uses:

```text
AITeamVN/Vietnamese_Reranker
```

The reranker receives query/chunk pairs and produces a relevance score.

Reranker scores are logits rather than normalized probabilities. Therefore:

```text
score A > score B
```

is meaningful for ranking, but a negative score is not an error.

The system should not remove chunks merely because two scores are equal.

---

# Context Builder

## 7. Legal Context Reconstruction

`ContextBuilder` transforms reranked chunks into structured legal context.

It:

1. Removes exact duplicates.
2. Groups chunks by document.
3. Groups chunks by article.
4. Sorts articles.
5. Sorts clauses.
6. Sorts points.
7. Preserves source metadata.
8. Renders an LLM-friendly context.

Example output:

```text
[VĂN BẢN]
Tiêu đề: Bộ luật Lao động 2019
Loại văn bản: Bộ luật
Cơ quan ban hành: Quốc hội
Ngày ban hành: ...

Điều 36. Quyền đơn phương chấm dứt hợp đồng lao động
Khoản 1
...

Khoản 2
...

Khoản 3
...
```

The builder keeps legal hierarchy visible to the LLM.

---

# Context Selector

## 8. Context Filtering

The selector reduces noise after reranking and context reconstruction.

The main scoring signals are:

```text
Reranker relevance
Lexical overlap
Issuance-date recency
Special-case awareness
```

The current design intentionally gives reranker relevance the highest weight.

An example configuration is conceptually:

```text
Reranker relevance       70%
Lexical overlap          15%
Issuance date             5%
Special-case awareness   10%
```

The exact weights remain configurable.

---

# Special-case Awareness

Legal questions frequently have exceptions or special populations.

For example, a query about ordinary employment termination should not automatically treat:

```text
người giúp việc gia đình
```

as the same legal case.

Likewise:

```text
ngành, nghề, công việc đặc thù
```

may introduce different notice periods.

The selector therefore distinguishes:

```text
special case mentioned in query
```

from:

```text
special case only mentioned in retrieved chunk
```

When a special case is not mentioned by the user, the selector can lower the relevance of that special-case chunk instead of deleting it unconditionally.

This is intentionally a soft filter.

---

# Issuance Date vs Effective Date

The system stores:

```text
issuance_date
```

This is the document's issuance date.

It must **not** automatically be interpreted as:

```text
effective_date
```

Therefore the selector may use `issuance_date` as a small ranking signal, but the system should not claim:

> "This is the currently effective law because it was issued later."

Legal validity and effective status require explicit legal metadata and/or source evidence.

---

# Answer Generation

## 9. Prompt Construction

The answer generator sends:

```text
System instructions
+
User question
+
Selected legal context
```

to the local Qwen3-4B model.

The system prompt instructs the model to:

- answer only from context;
- avoid inventing legal information;
- preserve the user's requested information;
- distinguish special cases;
- distinguish different documents/versions;
- cite only sources appearing in context;
- return insufficient evidence when the context is inadequate.

---

# Structured Answer Schema

The LLM returns a schema represented by:

```text
LegalAnswer
```

with fields:

```text
answer
citations
insufficient_evidence
note
```

Each citation contains:

```text
chunk_id
document_id
title
article
clause
point
issuance_date
```

The important addition is:

```text
chunk_id
```

because it allows a citation to be traced back to the exact retrieved evidence.

---

# Citation Flow

```text
User question
      ↓
Retrieved chunk
      ↓
chunk_id
      ↓
Context
      ↓
Qwen answer
      ↓
Citation
      ↓
chunk_id
      ↓
SQLite
      ↓
Original evidence
```

This makes it possible to implement source verification later.

---

# FastAPI Backend

The backend exposes the RAG pipeline through FastAPI.

## Health check

```http
GET /api/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Chat endpoint

```http
POST /api/chat
```

Request:

```json
{
  "query": "Người lao động đơn phương chấm dứt hợp đồng lao động thì phải báo trước bao nhiêu ngày?"
}
```

Response:

```json
{
  "query": "Người lao động đơn phương chấm dứt hợp đồng lao động thì phải báo trước bao nhiêu ngày?",
  "normalized_query": "...",
  "retrieved_count": 30,
  "reranked_count": 10,
  "selected_document_count": 3,
  "selected_chunk_count": 5,
  "answer": {
    "answer": "...",
    "citations": [
      {
        "chunk_id": "...",
        "document_id": 123,
        "title": "...",
        "article": "Điều 37",
        "clause": "Khoản 3",
        "point": null,
        "issuance_date": "..."
      }
    ],
    "insufficient_evidence": false,
    "note": null
  }
}
```

The exact numeric counts depend on the current retrieval result.

---

# Starting FastAPI

From the project root:

```powershell
uvicorn api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI documentation is useful for testing the API before starting the frontend.

---

# Next.js Frontend

The frontend is a separate Next.js application.

Expected structure:

```text
frontend/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
│
├── components/
│   ├── Chat.tsx
│   ├── ChatMessage.tsx
│   ├── CitationCard.tsx
│   └── DebugPanel.tsx
│
├── lib/
│   └── api.ts
│
├── public/
└── package.json
```

The frontend should communicate only with FastAPI.

It should **not** directly access:

- Qdrant;
- SQLite;
- the local LLM;
- Python model files.

---

# Frontend Environment

Create:

```text
frontend/.env.local
```

with:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

# Starting the Frontend

From the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

# CORS

During local development, FastAPI should allow the Next.js development origin:

```text
http://localhost:3000
```

and, if used:

```text
http://127.0.0.1:3000
```

For production, replace development CORS settings with the actual deployed frontend origin.

---

# Running the Complete System

The recommended development setup uses two terminals.

## Terminal 1: Qdrant

Ensure the Qdrant Docker container is running:

```powershell
docker ps
```

---

## Terminal 2: FastAPI

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

---

## Terminal 3: Next.js

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:3000
```

The browser request travels through:

```text
Next.js
  ↓
FastAPI
  ↓
LegalRAGPipeline
  ↓
Query Normalizer
  ↓
Retriever
  ↓
Qdrant
  ↓
Reranker
  ↓
Context Builder
  ↓
Context Selector
  ↓
Qwen3-4B
  ↓
LegalAnswer
```

---

# Running Individual Tests

The project contains both Python-direct executable tests and pytest-compatible test functions.

## Context Builder

Direct:

```powershell
python tests/test_context_builder.py
```

Pytest:

```powershell
pytest tests/test_context_builder.py -v
```

Expected result:

```text
RESULT: 14/14 tests passed
```

---

## Context Selector

```powershell
python tests/test_context_selector.py
```

or:

```powershell
pytest tests/test_context_selector.py -v
```

The selector tests include cases where:

- irrelevant special cases are penalized;
- explicitly requested special cases are retained;
- equal reranker scores do not imply duplicates;
- legal hierarchy remains ordered.

---

## Answer Schema

```powershell
python tests/test_answer_schema.py
```

Expected result in the current implementation:

```text
RESULT: 5/5 tests passed
```

---

## Answer Generator

```powershell
python tests/test_answer_generator.py
```

Expected result in the current implementation:

```text
RESULT: 3/3 tests passed
```

---

# Integration Testing

The project also contains integration-style tests that connect real pipeline stages.

For example:

```powershell
python tests/test_context_integration.py
```

A context integration test can show:

```text
Retriever
    ↓
Reranker
    ↓
Context Builder
    ↓
Final context
```

The most important part of such a test is not only `PASS/FAIL`, but the actual final context.

For legal RAG development, inspect whether:

- the correct article is present;
- the correct clause is present;
- the requested number/date/condition is preserved;
- historical documents are not overwhelming the context;
- special-case rules are clearly separated.

---

# Full RAG CLI

The project includes a single-entry pipeline runner intended to replace manually running each stage.

The preferred usage is:

```powershell
python scripts/run_rag.py "Người lao động đơn phương chấm dứt hợp đồng lao động thì phải báo trước bao nhiêu ngày?"
```

This should initialize and run:

```text
Embedding model
↓
Qdrant
↓
ChunkStore
↓
Local Qwen3-4B
↓
Query Normalizer
↓
Retriever
↓
Reranker
↓
Context Builder
↓
Context Selector
↓
Answer Generator
```

Application paths such as:

- model path;
- SQLite database path;
- Qdrant collection name;
- embedding dimension;

should be read from the project's `config.py` rather than duplicated in individual scripts.

---

# Example Questions

The following questions are useful for manually testing different parts of the retrieval stack.

## General legal question

```text
Người lao động đơn phương chấm dứt hợp đồng lao động thì phải báo trước bao nhiêu ngày?
```

This tests:

- legal topic retrieval;
- preservation of the "bao nhiêu ngày?" intent;
- general vs special-case filtering;
- citation generation.

## Colloquial phrasing

```text
Tôi muốn nghỉ việc ngay thì có phải báo trước không?
```

This tests query normalization of conversational language.

## Specific contract condition

```text
Hợp đồng lao động không xác định thời hạn, người lao động muốn nghỉ việc thì cần báo trước bao lâu?
```

This tests condition preservation.

## Special case

```text
Người giúp việc gia đình muốn nghỉ việc thì phải báo trước bao lâu?
```

This tests special-case awareness.

---

# Configuration

The project centralizes application configuration in:

```text
src/vn_legal_rag/config.py
```

Application code should import configuration values instead of duplicating them.

Examples of configuration that belong in the config layer include:

```text
model paths
embedding model name
embedding dimension
Qdrant collection
SQLite database path
retrieval top-k
rerank top-k
reranker settings
LLM settings
```

This is important because the same configuration is used by:

- CLI scripts;
- integration tests;
- FastAPI;
- the frontend-facing backend.

Avoid creating another copy of these paths/constants inside individual scripts.

---

# Development Workflow

A typical development cycle is:

```text
1. Modify one module
        ↓
2. Run its unit test
        ↓
3. Run integration test
        ↓
4. Run end-to-end CLI test
        ↓
5. Run FastAPI
        ↓
6. Test through Next.js UI
```

For example, when changing `ContextSelector`:

```powershell
python tests/test_context_selector.py
```

Then:

```powershell
python tests/test_context_integration.py
```

Then:

```powershell
python scripts/run_rag.py "..."
```

Finally test:

```text
http://localhost:3000
```

---

# Troubleshooting

## `python tests/test_*.py` produces no output

Make sure the script has a `__main__` block.

The current project test scripts use:

```python
if __name__ == "__main__":
    main()
```

Run them with:

```powershell
python tests/test_xxx.py
```

Not:

```powershell
tests/test_xxx.py
```

---

## `pytest` works but `python test_file.py` does nothing

This usually means the test file only defines `test_*` functions and does not call them directly.

Add a project-style `main()` test runner if direct execution is desired.

---

## `QdrantStore.__init__()` requires arguments

The current class requires:

```python
QdrantStore(
    collection_name=...,
    dimension=...,
)
```

Use the values defined by `config.py`.

Do not create another independent collection name in tests unless the test intentionally targets a separate collection.

---

## SQLite / ChunkStore errors

`ChunkStore` requires the configured database path.

The project uses Python's built-in `sqlite3` module, so SQLite itself does not need a separate Python package.

Verify that the path in `config.py` points to the generated ChunkStore database.

---

## `llama_cpp` import error

Verify:

```powershell
python -c "from llama_cpp import Llama; print('OK')"
```

If necessary, reinstall the CUDA wheel:

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

---

## Qwen model file not found

Expected location:

```text
models/Qwen3-4B-Q4_K_M.gguf
```

Check:

```powershell
dir models
```

and verify the configured model path.

---

## Hugging Face authentication warning

Messages such as:

```text
You are sending unauthenticated requests to the HF Hub
```

are warnings about Hugging Face access/rate limits.

The project can still work for public model downloads, but authenticated access may provide higher limits.

---

## Frontend cannot call FastAPI

Check:

```text
FastAPI:
http://127.0.0.1:8000

Next.js:
http://localhost:3000
```

Verify:

1. FastAPI is running.
2. `/api/health` responds.
3. `frontend/.env.local` points to the correct API.
4. FastAPI CORS allows the frontend origin.

---

## FastAPI works in Swagger but frontend fails

The RAG backend is probably working.

Inspect the browser developer console and verify:

```text
POST /api/chat
```

The frontend's `lib/api.ts` should use the same request/response schema exposed by FastAPI.

---

## Context contains many legal versions

This can happen when retrieval finds:

```text
historical laws
consolidated versions
special cases
current-looking documents
```

Do not solve this by blindly choosing the newest `issuance_date`.

Inspect:

```text
Retriever
↓
Reranker
↓
Context Selector
```

and verify that relevance, special-case conditions, and document differences are being handled correctly.

---

# Design Decisions

## Why Qdrant instead of keeping all vectors in Python?

The corpus is large enough that keeping the entire vector set in Python memory is undesirable.

Qdrant provides:

- persistent vector storage;
- vector search;
- metadata filtering;
- scalable index management;
- separation of vector storage from the Python process.

---

## Why SQLite as a separate text store?

The vector database is optimized for vector retrieval.

SQLite provides a lightweight local store for:

- complete source text;
- metadata;
- source information.

This keeps the retrieval layer efficient while retaining full evidence for answer generation.

---

## Why contextual embeddings?

Short legal chunks can be ambiguous without their hierarchy.

Embedding:

```text
article
+
clause
+
point
+
text
```

gives the embedding model more legal context than the raw fragment alone.

---

## Why both original and normalized query signals?

A query normalizer can improve clarity but can also accidentally remove important details.

The original query may contain information such as:

```text
bao nhiêu ngày
có được không
năm nào
trong trường hợp nào
```

The retrieval architecture therefore keeps the original query signal available rather than trusting a rewrite blindly.

---

## Why a reranker?

Vector similarity is useful for candidate generation, but it is not always sufficient for fine-grained legal relevance.

The reranker allows a stronger second-stage ordering:

```text
Vector retrieval
      ↓
broad candidate set
      ↓
Reranker
      ↓
smaller high-relevance set
```

---

## Why a Context Selector?

Even a good reranker can produce:

```text
correct rule
+
historical rule
+
special-case rule
+
related-but-not-answering rule
```

The Context Selector is a lightweight layer that reduces unnecessary context before a small local LLM sees it.

---

## Why use a local Qwen3-4B model?

The project is intended to support local/offline inference.

Advantages:

- no per-request external LLM API dependency;
- local control over prompts;
- local data handling;
- reproducible experiments.

Trade-offs:

- GPU memory constraints;
- generation latency;
- smaller reasoning capacity compared with much larger models.

---

# Current Limitations

The current system is a strong prototype, but several legal-specific problems remain open.

## Legal validity

The system currently has `issuance_date`, but that is not equivalent to legal effective status.

A future version should model:

```text
effective_date
expiry_date
replacement
repeal
partial amendment
legal status
```

when reliable source metadata is available.

## Citation verification

The answer schema now has `chunk_id`, which makes verification possible, but a final automatic verification layer has not yet been fully implemented.

A future verifier can ensure that every citation:

```text
exists in selected context
```

and optionally that the cited article/clause/point actually belongs to that chunk.

## Context expansion

Sometimes retrieval finds one chunk from an article but misses adjacent clauses that provide necessary conditions or exceptions.

A future context-expansion stage can retrieve neighboring chunks from the same legal article before generation.

## Streaming

The current local-generation workflow can behave as a synchronous request.

A future SSE/streaming implementation can report:

```text
normalizing
retrieving
reranking
building_context
generating
done
```

to the frontend.

## Evaluation dataset

The system has been manually tested with real legal questions, but a larger formal evaluation set is still recommended.

A later evaluation dataset should cover:

- general legal questions;
- conditions;
- dates;
- quantities;
- exceptions;
- colloquial phrasing;
- historical vs newer documents;
- questions with insufficient evidence.

---

# Roadmap

## Phase 1 — Data ingestion

- [x] Legal document ingestion
- [x] Chunk generation
- [x] Deterministic chunk IDs
- [x] ChunkStore
- [x] Qdrant indexing

## Phase 2 — Retrieval

- [x] multilingual-e5-base
- [x] Qdrant retrieval
- [x] Query normalization
- [x] Multi-query retrieval
- [x] Vietnamese reranking

## Phase 3 — Context

- [x] Context models
- [x] Context Builder
- [x] Hierarchical rendering
- [x] Exact deduplication
- [x] Context Selector
- [x] Special-case awareness

## Phase 4 — Answer Generation

- [x] Legal answer schema
- [x] Legal prompt
- [x] Local Qwen3-4B generation
- [x] Pydantic validation
- [x] Chunk-level citation IDs

## Phase 5 — Application

- [x] FastAPI `/api/health`
- [x] FastAPI `/api/chat`
- [x] Next.js / React frontend
- [x] Citation UI
- [x] Debug UI
- [ ] Streaming / SSE
- [ ] Citation verification
- [ ] Source viewer
- [ ] Conversation persistence

## Future RAG improvements

- [ ] Legal effective-date metadata
- [ ] Legal version / amendment graph
- [ ] Context expansion by article
- [ ] Better special-case detection
- [ ] Hybrid lexical + vector retrieval
- [ ] Retrieval evaluation metrics
- [ ] End-to-end answer evaluation
- [ ] Automatic hallucination checks

---

# End-to-End Quick Start

For a new development machine, the shortest high-level sequence is:

```powershell
# 1. Clone / enter project

# 2. Create environment
python -m venv .venv

# 3. Activate
.\.venv\Scripts\Activate.ps1

# 4. Install project
pip install -e .

# 5. Install CUDA llama.cpp wheel
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

# 6. Download local Qwen GGUF
hf download Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir models

# 7. Install Playwright browser if crawling features are needed
playwright install

# 8. Start Qdrant through Docker
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# 9. Start FastAPI
uvicorn api.main:app --reload
```

In a separate terminal:

```powershell
cd D:\Code\Vietnam-Legal-Chatbot\frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

# Development Principle

Keep the system modular.

The intended dependency direction is:

```text
Frontend
   ↓
API
   ↓
RAG Pipeline
   ↓
Retrieval / Context / Generation
   ↓
Models / Storage
```

The frontend should not know how Qdrant works.

The answer generator should not perform retrieval.

The retriever should not render UI.

The Context Builder should not decide legal validity.

Each layer should have one clear responsibility so that individual components can be tested and replaced independently.
