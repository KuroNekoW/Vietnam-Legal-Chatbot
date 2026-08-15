from pathlib import Path

# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ============================================================
# Documents
# ============================================================

LEGAL_DOCUMENT_FILE = (
    PROCESSED_DATA_DIR /
    "legal_documents.jsonl"
)

TOTAL_DOCUMENTS = 518_255

# ============================================================
# Chunk
# ============================================================

MAX_SUBCHUNKS_PER_ARTICLE = 100

CHUNK_DATA_DIR = (
    DATA_DIR /
    "chunks"
)

CHUNK_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_FILE = (
    CHUNK_DATA_DIR /
    "chunks.jsonl"
)

# ============================================================
# Embedding
# ============================================================

EMBEDDING_MODEL = (
    "intfloat/multilingual-e5-base"
)

EMBEDDING_BATCH_SIZE = 128

# ============================================================
# Vector Database (Qdrant)
# ============================================================

INDEX_DIR = (
    DATA_DIR /
    "index"
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

QDRANT_PATH = (
    INDEX_DIR /
    "qdrant_db"
)

QDRANT_COLLECTION = (
    "legal_documents"
)

# ============================================================
# Retrieval
# ============================================================

RETRIEVAL_DIR = DATA_DIR / "retrieval"

RETRIEVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_STORE_DB = (
    RETRIEVAL_DIR / "chunk_store.db"
)

# ============================================================
# LLM
# ============================================================

LLM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen3-4B-Q4_K_M.gguf"
)

LLM_CONTEXT_SIZE = 4096

LLM_GPU_LAYERS = -1

LLM_MAX_TOKENS = 512

LLM_TEMPERATURE = 0.0

# ============================================================
# Reranker
# ============================================================

RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"

RERANKER_DEVICE = "cuda"

RERANKER_BATCH_SIZE = 8

RERANKER_MAX_LENGTH = 2304