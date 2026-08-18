import queue
import threading
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
print("BUILD VECTOR DATABASE (OPTIMIZED MULTITHREADING)")
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

total_chunks = count_jsonl(CHUNK_FILE)
print(f"Total chunks    : {total_chunks:,}")

# ============================================================
# Current Qdrant Status
# ============================================================

already_indexed = store.ntotal
print(f"Already indexed : {already_indexed:,}")

remaining = max(total_chunks - already_indexed, 0)
print(f"Remaining       : {remaining:,}")
print()

# ============================================================
# Builder & Queue Setup
# ============================================================

builder = IndexBuilder(
    embedding_model=model,
    vector_store=store,
    batch_size=EMBEDDING_BATCH_SIZE,
)

# Khởi tạo Queue giới hạn kích thước để tránh tràn RAM nếu CPU đọc quá nhanh
batch_queue = queue.Queue(maxsize=10)

# ============================================================
# Progress Bars
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
# Producer Thread (Data Preparation)
# ============================================================

def data_producer():
    """
    Luồng phụ: Chuyên đọc file, gom batch và kiểm tra missing.
    """
    batch = []
    try:
        for chunk in load_chunks_jsonl(CHUNK_FILE):
            batch.append(chunk)
            scan_progress.update(1)

            if len(batch) < builder.batch_size:
                continue

            missing = store.filter_missing(batch)
            if missing:
                batch_queue.put(missing) # Nhét batch vào hàng đợi cho GPU
            batch.clear()

        # Xử lý batch cuối cùng (nếu còn dư)
        if batch:
            missing = store.filter_missing(batch)
            if missing:
                batch_queue.put(missing)
            batch.clear()

    except Exception as e:
        print(f"\nProducer Error: {e}")
    finally:
        # Gửi tín hiệu kết thúc (None) để dừng luồng chính
        batch_queue.put(None)

# Bắt đầu luồng Producer
producer_thread = threading.Thread(target=data_producer, daemon=True)
producer_thread.start()

# ============================================================
# Consumer Thread (Main Thread - GPU Encoding)
# ============================================================

interrupted = False

try:
    while True:
        # Đợi và lấy batch đã được lọc từ hàng đợi
        missing_batch = batch_queue.get()

        # Nhận được tín hiệu kết thúc từ Producer
        if missing_batch is None:
            break

        # GPU thực hiện encode và insert
        builder.process_batch(missing_batch)
        index_progress.update(len(missing_batch))
        
        # Báo cho queue biết đã xử lý xong tác vụ này
        batch_queue.task_done()

except KeyboardInterrupt:
    interrupted = True
    print("\n\n" + "=" * 60)
    print("BUILD INTERRUPTED")
    print("=" * 60 + "\n")
    print("Vectors already submitted to Qdrant remain stored.")
    print("Run this script again to resume.")

finally:
    scan_progress.close()
    index_progress.close()

# ============================================================
# Final Qdrant Status
# ============================================================

final_count = store.ntotal
missing_count = max(total_chunks - final_count, 0)

print("\n" + "=" * 60)

if interrupted:
    print("BUILD INTERRUPTED")
elif final_count >= total_chunks:
    print("BUILD FINISHED")
else:
    print("BUILD INCOMPLETE")

print("=" * 60 + "\n")
print(f"Total chunks    : {total_chunks:,}")
print(f"Vectors stored  : {final_count:,}")
print(f"Vectors missing : {missing_count:,}")
print(f"Collection      : {QDRANT_COLLECTION}\n")

if not interrupted and missing_count > 0:
    print("Some chunks are still missing from Qdrant.")
    print("Run this script again to resume.\n")