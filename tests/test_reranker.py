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


# ============================================================
# Config
# ============================================================

QUERY = (
    "Đơn phương chấm dứt hợp đồng lao động, người lao động phải báo trước bao nhiêu ngày?"
)

CANDIDATE_K = 30

RERANK_TOP_K = 8


# ============================================================
# Main
# ============================================================

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

print()


# ============================================================
# Normalize query
# ============================================================

print("-" * 70)
print("QUERY NORMALIZATION")
print("-" * 70)

print()
print("Original query:")
print(QUERY)

normalized = normalizer.normalize(
    QUERY
)

normalized_query = (
    normalized.normalized_query
)

print()
print("Normalized query:")
print(normalized_query)

print()
print("Question intent:")

print(
    f"  {normalized.question_intent}"
)

print()

print("Keywords:")

for keyword in normalized.keywords:

    print(
        f"  - {keyword}"
    )

print()

print("Constraints:")

for constraint in normalized.constraints:

    print(
        f"  - {constraint}"
    )

print()

print("Temporal constraints:")

for temporal in normalized.temporal_constraints:

    print(
        f"  - {temporal}"
    )

print()


# ============================================================
# Embedding
# ============================================================

print("-" * 70)
print("EMBEDDING MODEL")
print("-" * 70)

print()
print("Loading embedding model...")

embedding_model = EmbeddingModel()

print()
print(
    f"Device    : {embedding_model.device}"
)

print(
    f"Dimension : {embedding_model.dimension}"
)

print()


# ============================================================
# Qdrant
# ============================================================

print("-" * 70)
print("QDRANT")
print("-" * 70)

print()
print("Connecting Qdrant...")

qdrant = QdrantStore(
    collection_name=QDRANT_COLLECTION,
    dimension=embedding_model.dimension,
    database_path=QDRANT_PATH,
)

print()
print(
    f"Vectors : {qdrant.ntotal:,}"
)

print()


# ============================================================
# Chunk Store
# ============================================================

print("-" * 70)
print("CHUNK STORE")
print("-" * 70)

print()

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

print("-" * 70)
print("RETRIEVAL")
print("-" * 70)

print()
print("Running retrieval...")

candidates = retriever.retrieve(
    query=QUERY,
    top_k=CANDIDATE_K,
    candidate_k=CANDIDATE_K,
)

print()

print(
    f"Candidates : {len(candidates)}"
)

print()


# ============================================================
# BEFORE RERANK
# ============================================================

print()
print("=" * 70)
print("BEFORE RERANK")
print("=" * 70)
print()

for i, chunk in enumerate(
    candidates,
    start=1,
):

    print(
        f"#{i:02d}"
    )

    print(
        f"Qdrant score : "
        f"{chunk.score:.6f}"
    )

    print(
        f"Source query : "
        f"{chunk.source_query}"
    )

    print(
        f"Chunk ID     : "
        f"{chunk.chunk_id}"
    )

    print(
        f"Document ID  : "
        f"{chunk.document_id}"
    )

    print(
        f"Article      : "
        f"{chunk.article_no}"
    )

    print(
        f"Article      : "
        f"{chunk.article}"
    )

    print(
        f"Clause       : "
        f"{chunk.clause_no}"
    )

    print(
        f"Clause       : "
        f"{chunk.clause}"
    )

    print(
        f"Point        : "
        f"{chunk.point_no}"
    )

    print(
        f"Point        : "
        f"{chunk.point}"
    )

    print(
        f"Title        : "
        f"{chunk.title}"
    )

    preview = (
        chunk.text
        .replace("\n", " ")
        .strip()
    )

    if len(preview) > 400:

        preview = (
            preview[:400]
            + "..."
        )

    print(
        f"Text         : "
        f"{preview}"
    )

    print(
        "-" * 70
    )


# ============================================================
# Reranker
# ============================================================

print()
print("=" * 70)
print("RERANKER")
print("=" * 70)
print()

reranker = Reranker()

print()

print(
    "Reranking with normalized query..."
)

results = reranker.rerank(
    query=normalized_query,
    chunks=candidates,
    top_k=RERANK_TOP_K,
)


# ============================================================
# AFTER RERANK
# ============================================================

print()
print("=" * 70)
print("AFTER RERANK")
print("=" * 70)
print()

print(
    f"Results : {len(results)}"
)

print()

for i, chunk in enumerate(
    results,
    start=1,
):

    print(
        f"#{i}"
    )

    print(
        f"Rerank score : "
        f"{chunk.rerank_score:.6f}"
    )

    print(
        f"Qdrant score : "
        f"{chunk.score:.6f}"
    )

    print(
        f"Source query : "
        f"{chunk.source_query}"
    )

    print(
        f"Chunk ID     : "
        f"{chunk.chunk_id}"
    )

    print(
        f"Document ID  : "
        f"{chunk.document_id}"
    )

    print(
        f"Article no   : "
        f"{chunk.article_no}"
    )

    print(
        f"Article      : "
        f"{chunk.article}"
    )

    print(
        f"Clause no    : "
        f"{chunk.clause_no}"
    )

    print(
        f"Clause       : "
        f"{chunk.clause}"
    )

    print(
        f"Point no     : "
        f"{chunk.point_no}"
    )

    print(
        f"Point        : "
        f"{chunk.point}"
    )

    print(
        f"Title        : "
        f"{chunk.title}"
    )

    print()

    print(
        chunk.text
    )

    print()
    print("-" * 70)


# ============================================================
# Deduplication report
# ============================================================

duplicate_groups = (
    reranker.last_duplicate_groups
)

print()
print("=" * 70)
print("DEDUPLICATION REPORT")
print("=" * 70)
print()

print(
    f"Duplicate groups removed : "
    f"{len(duplicate_groups)}"
)

print()

for i, group in enumerate(
    duplicate_groups,
    start=1,
):

    kept = group["kept"]
    removed = group["removed"]

    print(
        f"GROUP #{i}"
    )

    print(
        f"Kept    : "
        f"{kept.document_id} | "
        f"{kept.title}"
    )

    print(
        f"Removed : "
        f"{removed.document_id} | "
        f"{removed.title}"
    )

    print(
        f"Article no: "
        f"{kept.article_no}"
    )

    print(
        f"Article   : "
        f"{kept.article}"
    )

    print(
        f"Clause no : "
        f"{kept.clause_no}"
    )

    print(
        f"Clause    : "
        f"{kept.clause}"
    )

    print(
        f"Point no  : "
        f"{kept.point_no}"
    )

    print(
        f"Point     : "
        f"{kept.point}"
    )

    print(
        f"Issuance date    : "
        f"{chunk.issuance_date}"
    )

    print()

    print(
        "Text:"
    )

    print(
        kept.text
    )

    print(
        "-" * 70
    )

# ============================================================
# Cleanup
# ============================================================

chunk_store.close()

print()
print("=" * 70)
print("TEST FINISHED")
print("=" * 70)
print()