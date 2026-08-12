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


# ============================================================
# Embedding Model
# ============================================================

print("Loading embedding model...")

model = EmbeddingModel()

print(f"Device      : {model.device}")
print(f"Dimension   : {model.dimension}")
print()


# ============================================================
# Qdrant
# ============================================================

print("Connecting Qdrant...")

store = QdrantStore(
    collection_name=QDRANT_COLLECTION,
    dimension=model.dimension,
    database_path=QDRANT_PATH,
)

print()


# ============================================================
# Dataset
# ============================================================

total_chunks = count_jsonl(
    CHUNK_FILE
)

print(
    f"Total chunks    : {total_chunks:,}"
)

# ============================================================
# Current Qdrant Status
# ============================================================

already_indexed = store.ntotal

print(
    f"Already indexed : {already_indexed:,}"
)

remaining = max(
    total_chunks - already_indexed,
    0,
)

print(
    f"Remaining       : {remaining:,}"
)

print()


# ============================================================
# Builder
# ============================================================

builder = IndexBuilder(
    embedding_model=model,
    vector_store=store,
    batch_size=EMBEDDING_BATCH_SIZE,
)


# ============================================================
# Batch
# ============================================================

batch = []


# ============================================================
# Progress Bars
# ============================================================
#
# Scanning:
#   Số chunk đã đọc từ chunks.jsonl.
#
# Indexed:
#   Số vector đã được index theo số chunk mới phát hiện.
#
# Indexed bắt đầu từ số vector Qdrant hiện có.
#
# ============================================================

scan_progress = tqdm(
    total=total_chunks,
    initial=0,
    desc="Scanning",
    unit="chunk",
    colour="blue",
    dynamic_ncols=True,
)

index_progress = tqdm(
    total=total_chunks,
    initial=already_indexed,
    desc="Indexed ",
    unit="chunk",
    colour="green",
    dynamic_ncols=True,
)


# ============================================================
# Build
# ============================================================

interrupted = False

try:

    for chunk in load_chunks_jsonl(
        CHUNK_FILE
    ):

        batch.append(chunk)

        # ----------------------------------------------------
        # Scanning progress
        # ----------------------------------------------------

        scan_progress.update(1)

        # ----------------------------------------------------
        # Batch chưa đầy
        # ----------------------------------------------------

        if len(batch) < builder.batch_size:
            continue

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------
        #
        # Chỉ lấy những chunk chưa tồn tại trong Qdrant.
        #
        # Không load toàn bộ chunk_id vào RAM.
        #

        missing = store.filter_missing(
            batch
        )

        # ----------------------------------------------------
        # Encode + Insert
        # ----------------------------------------------------

        if missing:

            builder.process_batch(
                missing
            )

            # Chỉ tính chunk thực sự mới.
            index_progress.update(
                len(missing)
            )

        # ----------------------------------------------------
        # Clear batch
        # ----------------------------------------------------

        batch.clear()


    # ========================================================
    # Remaining Batch
    # ========================================================

    if batch:

        missing = store.filter_missing(
            batch
        )

        if missing:

            builder.process_batch(
                missing
            )

            index_progress.update(
                len(missing)
            )

        batch.clear()


except KeyboardInterrupt:

    interrupted = True

    print()
    print()
    print("=" * 60)
    print("BUILD INTERRUPTED")
    print("=" * 60)
    print()

    print(
        "Vectors already submitted to Qdrant remain stored."
    )

    print(
        "Run this script again to resume."
    )

finally:

    scan_progress.close()
    index_progress.close()


# ============================================================
# Final Qdrant Status
# ============================================================

final_count = store.ntotal

missing_count = max(
    total_chunks - final_count,
    0,
)

print()
print("=" * 60)


# ============================================================
# Interrupted
# ============================================================

if interrupted:

    print("BUILD INTERRUPTED")


# ============================================================
# Complete
# ============================================================

elif final_count >= total_chunks:

    print("BUILD FINISHED")


# ============================================================
# Incomplete
# ============================================================

else:

    print("BUILD INCOMPLETE")


print("=" * 60)
print()

print(
    f"Total chunks    : {total_chunks:,}"
)

print(
    f"Vectors stored  : {final_count:,}"
)

print(
    f"Vectors missing : {missing_count:,}"
)

print(
    f"Collection      : {QDRANT_COLLECTION}"
)

print()


# ============================================================
# Resume Message
# ============================================================

if not interrupted and missing_count > 0:

    print(
        "Some chunks are still missing from Qdrant."
    )

    print(
        "Run this script again to resume."
    )

    print()