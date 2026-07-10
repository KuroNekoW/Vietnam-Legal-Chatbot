from pathlib import Path

from vn_legal_rag.config import (
    CHUNK_FILE,
    CHUNK_INDEX_FILE,
    FAISS_INDEX_FILE,
    EMBEDDING_BATCH_SIZE,
)

from vn_legal_rag.embedding import (
    EmbeddingModel,
    EmbeddingPipeline,
)

from vn_legal_rag.retrieval import (
    FaissIndex,
)

from vn_legal_rag.utils import (
    load_chunks_jsonl,
)


def main():

    print()
    print("=" * 60)
    print("EXPORT EMBEDDINGS")
    print("=" * 60)
    print()

    print("Starting...")
    #
    # Model
    #

    print()
    print("Loading embedding model...")

    model = EmbeddingModel()

    print(f"Device     : {model.device}")
    print(f"Dimension  : {model.dimension}")

    #
    # FAISS
    #

    faiss_index = FaissIndex(
        dimension=model.dimension,
    )

    #
    # Remove old chunk index
    #

    if CHUNK_INDEX_FILE.exists():

        CHUNK_INDEX_FILE.unlink()

    #
    # Pipeline
    #

    pipeline = EmbeddingPipeline(

        model=model,

        faiss_index=faiss_index,

        chunk_index_path=CHUNK_INDEX_FILE,

        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print()
    print("Loading chunks...")

    chunks = load_chunks_jsonl(
        CHUNK_FILE
    )

    total_chunks = 11334109

    pipeline.export(

        chunks,

        total_chunks=total_chunks,
    )

    print()
    print("Saving FAISS index...")

    faiss_index.save(
        FAISS_INDEX_FILE
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()

    print(f"Vectors : {faiss_index.size:,}")

    print()

    print("Saved")

    print(FAISS_INDEX_FILE)

    print(CHUNK_INDEX_FILE)


if __name__ == "__main__":

    main()