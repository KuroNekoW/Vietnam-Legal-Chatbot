from tqdm import tqdm

from vn_legal_rag.config import (
    CHUNK_FILE,
    EMBEDDING_BATCH_SIZE,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

from vn_legal_rag.embedding import EmbeddingModel

from vn_legal_rag.retrieval import (
    QdrantStore,
    IndexBuilder,
)

from vn_legal_rag.utils import (
    load_chunks_jsonl,
    count_jsonl,
)


# ============================================================
# Main
# ============================================================

print()
print("=" * 60)
print("BUILD VECTOR DATABASE")
print("=" * 60)
print()

# ------------------------------------------------------------
# Embedding Model
# ------------------------------------------------------------

print("Loading embedding model...")

model = EmbeddingModel()

print(f"Device      : {model.device}")
print(f"Dimension   : {model.dimension}")
print()

# ------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------

print("Connecting Qdrant...")

store = QdrantStore(
    collection_name=QDRANT_COLLECTION,
    dimension=model.dimension,
    database_path=QDRANT_PATH,
)

print("Loading indexed chunk ids...")

indexed_chunk_ids = store.load_existing_chunk_ids()

print(f"Already indexed : {len(indexed_chunk_ids):,}")
print()

# ------------------------------------------------------------
# Builder
# ------------------------------------------------------------

builder = IndexBuilder(
    embedding_model=model,
    vector_store=store,
    batch_size=EMBEDDING_BATCH_SIZE,
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

total_chunks = count_jsonl(CHUNK_FILE)

remaining = total_chunks - len(indexed_chunk_ids)

print(f"Total chunks : {total_chunks:,}")
print(f"Remaining    : {remaining:,}")
print()

batch = []

progress = tqdm(
    total=total_chunks,
    initial=len(indexed_chunk_ids),
    desc="Indexing",
    unit="chunk",
    colour="green",
    dynamic_ncols=True,
)

try:

    for chunk in load_chunks_jsonl(CHUNK_FILE):

        #
        # Resume
        #

        if chunk.chunk_id in indexed_chunk_ids:
            continue

        batch.append(chunk)

        #
        # Batch full
        #

        if len(batch) >= builder.batch_size:

            builder.process_batch(batch)

            indexed_chunk_ids.update(
                c.chunk_id
                for c in batch
            )

            progress.update(
                len(batch)
            )

            batch.clear()

    #
    # Remaining
    #

    if batch:

        builder.process_batch(batch)

        indexed_chunk_ids.update(
            c.chunk_id
            for c in batch
        )

        progress.update(
            len(batch)
        )

except KeyboardInterrupt:

    print()
    print("Interrupted by user.")
    print("All processed vectors have already been stored.")
    print("Run this script again to resume.")

finally:

    progress.close()

# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print()
print("=" * 60)
print("BUILD FINISHED")
print("=" * 60)
print()

print(f"Vectors stored : {builder.vectors:,}")
print(f"Collection     : {QDRANT_COLLECTION}")

print()