from vn_legal_rag.config import (
    CHUNK_STORE_DB,
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
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
)


# ============================================================
# Config
# ============================================================

QUERY = (
    "tôi mua đất giấy tay năm 2015 "
    "giờ làm sổ đỏ được không"
)

TOP_K = 10

CANDIDATE_K = 20


# ============================================================
# Main
# ============================================================

print()
print("=" * 70)
print("TEST RETRIEVAL")
print("=" * 70)
print()


# ============================================================
# Local LLM
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
    llm=llm
)

print()


# ============================================================
# Query normalization
# ============================================================

print(
    "Original query:"
)

print(
    QUERY
)

print()

normalized = normalizer.normalize(
    QUERY
)

print(
    "Normalized query:"
)

print(
    normalized.normalized_query
)

print()

print(
    "Legal terms:"
)

for term in normalized.legal_terms:

    print(
        f"  - {term}"
    )

print()

print(
    "Constraints:"
)

for constraint in normalized.constraints:

    print(
        f"  - {constraint}"
    )

print()


# ============================================================
# Embedding Model
# ============================================================

print("Loading embedding model...")

embedding_model = EmbeddingModel()

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

print("Opening chunk store...")

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

print(
    "Running retrieval..."
)

print()

results = retriever.retrieve(

    query=QUERY,

    top_k=TOP_K,

    candidate_k=CANDIDATE_K,

)


# ============================================================
# Results
# ============================================================

print(
    f"Retrieved : {len(results)}"
)

print()


for i, result in enumerate(
    results,
    start=1,
):

    print("=" * 70)

    print(
        f"#{i}"
    )

    print(
        f"Score       : {result.score:.6f}"
    )

    print(
        f"Source query: {result.source_query}"
    )

    print(
        f"Chunk ID    : {result.chunk_id}"
    )

    print(
        f"Document ID : {result.document_id}"
    )

    print(
        f"Title       : {result.title}"
    )

    print(
        f"Article     : {result.article_no}"
    )

    print(
        f"Clause      : {result.clause_no}"
    )

    print(
        f"Point       : {result.point_no}"
    )

    print(
        f"Chunk index : {result.chunk_index}"
    )

    print(
        f"Sub chunk   : {result.sub_chunk_index}"
    )

    print()

    print(
        result.text
    )

    print()


# ============================================================
# Cleanup
# ============================================================

chunk_store.close()

print("=" * 70)
print("TEST FINISHED")
print("=" * 70)
print()