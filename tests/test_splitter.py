from vn_legal_rag.chunking import LegalChunker
from vn_legal_rag.config import PROCESSED_DATA_DIR
from vn_legal_rag.utils import load_jsonl


documents = load_jsonl(
    PROCESSED_DATA_DIR / "legal_documents.jsonl"
)

doc = next(documents)

chunker = LegalChunker()

chunks = list(
    chunker.chunk(doc)
)

print()

print(f"Total chunks: {len(chunks)}")

print()

for chunk in chunks:

    print("=" * 100)

    print(chunk.chunk_id)

    print(chunk.article)

    print(chunk.start_char)

    print(chunk.end_char)

    print()

    print(chunk.text[:500])

    print()