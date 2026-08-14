from tqdm import tqdm

from vn_legal_rag.config import (
    CHUNK_FILE,
    CHUNK_STORE_DB,
)

from vn_legal_rag.retrieval import ChunkStore

from vn_legal_rag.utils import count_jsonl


print()
print("=" * 60)
print("BUILD CHUNK STORE")
print("=" * 60)
print()

print(
    f"Source : {CHUNK_FILE}"
)

print(
    f"Target : {CHUNK_STORE_DB}"
)

print()

# ============================================================
# Count chunks
# ============================================================

print("Counting chunks...")

total_chunks = count_jsonl(
    CHUNK_FILE
)

print(
    f"Total chunks : {total_chunks:,}"
)

print()

# ============================================================
# SQLite
# ============================================================

store = ChunkStore(
    CHUNK_STORE_DB
)

print(
    "Building SQLite chunk store..."
)

print()

# ============================================================
# Build
# ============================================================

store.build_from_jsonl(
    CHUNK_FILE,
    batch_size=5000,
    total=total_chunks,
)

# ============================================================
# Finish
# ============================================================

print()
print("=" * 60)
print("BUILD FINISHED")
print("=" * 60)
print()

print(
    f"Chunks stored : {store.count:,}"
)

print(
    f"Database      : {CHUNK_STORE_DB}"
)

print()

store.close()