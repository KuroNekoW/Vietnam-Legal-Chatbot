from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

LEGAL_DOCUMENT_FILE = PROCESSED_DATA_DIR / "legal_documents.jsonl"

EMBEDDING_DIR = DATA_DIR / "embeddings"

TOTAL_DOCUMENTS = 518255

# Chunks

MAX_SUBCHUNKS_PER_ARTICLE = 100

CHUNK_DATA_DIR = DATA_DIR / "chunks"

CHUNK_DATA_DIR.mkdir(parents=True, exist_ok=True,)

CHUNK_FILE = CHUNK_DATA_DIR / "chunks.jsonl"

# Index
INDEX_DIR = DATA_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True,)

FAISS_INDEX_FILE = (INDEX_DIR / "index.faiss")

CHUNK_INDEX_FILE = (INDEX_DIR / "chunk_index.jsonl")

# Embedding

EMBEDDING_MODEL = "BAAI/bge-m3"

EMBEDDING_BATCH_SIZE = 1024

EMBEDDINGS_PATH = (PROCESSED_DATA_DIR /"embeddings.npy")

CHUNK_INDEX_PATH = (PROCESSED_DATA_DIR /"chunk_index.json")