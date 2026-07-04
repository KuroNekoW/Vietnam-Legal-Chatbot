from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

LEGAL_DOCUMENT_FILE = PROCESSED_DATA_DIR / "legal_documents.jsonl"

EMBEDDING_DIR = DATA_DIR / "embeddings"

TOTAL_DOCUMENTS = 518255

# Chunks

CHUNK_DATA_DIR = DATA_DIR / "chunks"

CHUNK_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_FILE = CHUNK_DATA_DIR / "chunks.jsonl"