from vn_legal_rag.config import PROCESSED_DATA_DIR

from vn_legal_rag.utils import load_jsonl

from vn_legal_rag.chunking import (
    LegalSplitter,
    LengthSplitter,
)


def main():

    # Load first document
    documents = load_jsonl(
        PROCESSED_DATA_DIR / "legal_documents.jsonl"
    )

    document = next(documents)

    # Step 1: Legal-aware splitting
    legal_chunks = list(
        LegalSplitter().split(document)
    )

    # Step 2: Length-aware splitting
    splitter = LengthSplitter(
        chunk_size=1000,
        overlap=200,
    )

    final_chunks = []

    for chunk in legal_chunks:
        final_chunks.extend(
            splitter.split(chunk)
        )

    # Summary
    print()

    print("=" * 100)
    print(document.title)
    print("=" * 100)

    print()

    print(f"Semantic chunks : {len(legal_chunks)}")
    print(f"Final chunks    : {len(final_chunks)}")

    # Detail
    for i, chunk in enumerate(final_chunks):

        print()
        print("=" * 100)
        print(f"Chunk {i}")
        print("=" * 100)

        print(f"chunk_id           : {chunk.chunk_id}")
        print(f"document_id        : {chunk.document_id}")

        print()

        print(f"title              : {chunk.title}")
        print(f"article            : {chunk.article}")

        print()

        print(f"legal_type         : {chunk.legal_type}")
        print(f"legal_sectors      : {chunk.legal_sectors}")
        print(f"issuing_authority  : {chunk.issuing_authority}")
        print(f"issuance_date      : {chunk.issuance_date}")

        print()

        print(f"chunk_index        : {chunk.chunk_index}")
        print(f"start_char         : {chunk.start_char}")
        print(f"end_char           : {chunk.end_char}")
        print(f"length             : {len(chunk.text)}")

        print()

        print("-" * 100)
        print(chunk.text[:700])

        if len(chunk.text) > 700:
            print("...")

        print("-" * 100)


if __name__ == "__main__":
    main()