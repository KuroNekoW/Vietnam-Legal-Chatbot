from pathlib import Path

from tqdm import tqdm

from vn_legal_rag.config import (
    CHUNK_FILE,
    CHUNK_INDEX_FILE,
    FAISS_INDEX_FILE,
    EMBEDDING_BATCH_SIZE,
)

from vn_legal_rag.embedding import EmbeddingModel

from vn_legal_rag.retrieval import (
    FaissIndex,
    IndexBuilder,
)

from vn_legal_rag.utils import load_chunks_jsonl


# ============================================================
# Config
# ============================================================

CHECKPOINT_EVERY = 100_000


# ============================================================
# Main
# ============================================================

print()
print("=" * 60)
print("BUILD VECTOR INDEX")
print("=" * 60)
print()

print("Loading embedding model...")

model = EmbeddingModel()

print("Device     :", model.device)
print("Dimension  :", model.dimension)
print()

print("Creating FAISS index...")

faiss_index = FaissIndex(
    dimension=model.dimension,
)

# overwrite metadata file
Path(CHUNK_INDEX_FILE).parent.mkdir(
    parents=True,
    exist_ok=True,
)

Path(CHUNK_INDEX_FILE).write_text(
    "",
    encoding="utf-8",
)

builder = IndexBuilder(
    embedding_model=model,
    faiss_index=faiss_index,
    chunk_index_path=CHUNK_INDEX_FILE,
)

print("Loading chunks...")
print()

batch = []

processed = 0

progress = tqdm(
    desc="Indexing",
    unit="chunk",
    colour="green",
    dynamic_ncols=True,
)

for chunk in load_chunks_jsonl(
    CHUNK_FILE,
):

    batch.append(
        chunk
    )

    if len(batch) >= EMBEDDING_BATCH_SIZE:

        processed += builder.process_batch(
            batch,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        progress.update(
            len(batch)
        )

        batch.clear()

        #
        # checkpoint
        #

        if processed % CHECKPOINT_EVERY == 0:

            print()

            print(
                f"[Checkpoint] Saving index ({processed:,} vectors)..."
            )

            faiss_index.save(
                FAISS_INDEX_FILE
            )

            print("Done.")
            print()

#
# remaining
#

if batch:

    processed += builder.process_batch(
        batch,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    progress.update(
        len(batch)
    )

progress.close()

print()
print("Saving final index...")

faiss_index.save(
    FAISS_INDEX_FILE
)

print()

print("=" * 60)
print("BUILD FINISHED")
print("=" * 60)
print()

print(f"Vectors : {faiss_index.ntotal:,}")
print(f"Index   : {FAISS_INDEX_FILE}")
print(f"Metadata: {CHUNK_INDEX_FILE}")

print()