from vn_legal_rag.config import (
    CHUNK_STORE_DB,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_TEMPERATURE,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

from vn_legal_rag.embedding import EmbeddingModel

from vn_legal_rag.llm import LocalLLM

from vn_legal_rag.query import QueryNormalizer

from vn_legal_rag.retrieval import (
    ChunkStore,
    QdrantStore,
    Retriever,
    Reranker,
)


QUERY = (
    "tôi mua đất giấy tay năm 2015 "
    "giờ làm sổ đỏ được không"
)

CANDIDATE_K = 30
RERANK_TOP_K = 8


print()
print("=" * 70)
print("TEST RERANKER")
print("=" * 70)
print()


# ============================================================
# Query Normalizer
# ============================================================

print("Loading local LLM...")

llm = LocalLLM(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_CONTEXT_SIZE,
    n_gpu_layers=LLM_GPU_LAYERS,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
)

normalizer = QueryNormalizer(
    llm
)


# ============================================================
# Embedding
# ============================================================

print()
print("Loading embedding model...")

embedding_model = EmbeddingModel()

print()


# ============================================================
# Qdrant
# ============================================================

print("Connecting Qdrant...")

qdrant = QdrantStore(
    collection_name=QDRANT_COLLECTION,
    dimension=embedding_model.dimension,
    database_path=QDRANT_PATH,
)

print(
    f"Vectors : {qdrant.ntotal:,}"
)

print()


# ============================================================
# Chunk Store
# ============================================================

chunk_store = ChunkStore(
    CHUNK_STORE_DB
)

print(
    f"Chunks : {chunk_store.count:,}"
)

print()


# ============================================================
# Retriever
# ============================================================

retriever = Retriever(
    embedding_model=embedding_model,
    vector_store=qdrant,
    chunk_store=chunk_store,
    query_normalizer=normalizer,
)


# ============================================================
# Retrieval
# ============================================================

print("Running retrieval...")

candidates = retriever.retrieve(
    query=QUERY,
    top_k=CANDIDATE_K,
    candidate_k=CANDIDATE_K,
)

print(
    f"Candidates : {len(candidates)}"
)

print()


# ============================================================
# Reranker
# ============================================================

reranker = Reranker()

print()

print("Running reranking...")

results = reranker.rerank(
    query=QUERY,
    chunks=candidates,
    top_k=RERANK_TOP_K,
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("RERANKED RESULTS")
print("=" * 70)
print()

for i, chunk in enumerate(
    results,
    start=1,
):

    print("-" * 70)

    print(
        f"#{i}"
    )

    print(
        f"Qdrant score   : "
        f"{chunk.score:.6f}"
    )

    print(
        f"Rerank score   : "
        f"{chunk.rerank_score:.6f}"
    )

    print(
        f"Chunk ID       : "
        f"{chunk.chunk_id}"
    )

    print(
        f"Document ID    : "
        f"{chunk.document_id}"
    )

    print(
        f"Article        : "
        f"{chunk.article_no}"
    )

    print(
        f"Clause         : "
        f"{chunk.clause_no}"
    )

    print(
        f"Point          : "
        f"{chunk.point_no}"
    )

    print(
        f"Title          : "
        f"{chunk.title}"
    )

    print()

    print(
        chunk.text
    )

    print()


chunk_store.close()

print("=" * 70)
print("TEST FINISHED")
print("=" * 70)