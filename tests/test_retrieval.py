from vn_legal_rag.config import (
    QDRANT_COLLECTION,
    QDRANT_PATH,
    CHUNK_STORE_DB,
)

from vn_legal_rag.embedding import EmbeddingModel

from vn_legal_rag.retrieval import (
    QdrantStore,
    ChunkStore,
    Retriever,
)


# ============================================================
# Query
# ============================================================

QUERY = (
    "Cố ý gây thương tích bao nhiêu % là đi tù"
)

TOP_K = 10


# ============================================================
# Embedding
# ============================================================

print()
print("=" * 70)
print("TEST RETRIEVAL")
print("=" * 70)
print()

print("Loading embedding model...")

model = EmbeddingModel()

print(
    f"Device    : {model.device}"
)

print(
    f"Dimension : {model.dimension}"
)

print()


# ============================================================
# Qdrant
# ============================================================

print("Connecting Qdrant...")

qdrant = QdrantStore(
    collection_name=QDRANT_COLLECTION,
    dimension=model.dimension,
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
    embedding_model=model,
    vector_store=qdrant,
    chunk_store=chunk_store,
)


# ============================================================
# Search
# ============================================================

print(
    f"Query: {QUERY}"
)

print()

results = retriever.retrieve(
    query=QUERY,
    top_k=TOP_K,
)


# ============================================================
# Results
# ============================================================

print(
    f"Retrieved: {len(results)}"
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

    print()

    print(
        result.text
    )

    print()


# ============================================================
# Close
# ============================================================

chunk_store.close()

print("=" * 70)
print("DONE")
print("=" * 70)