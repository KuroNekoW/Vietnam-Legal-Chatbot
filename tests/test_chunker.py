from vn_legal_rag.chunking import (
    LegalSplitter,
    LengthSplitter,
)

from vn_legal_rag.config import (
    LEGAL_DOCUMENT_FILE,
)

from vn_legal_rag.utils import (
    load_jsonl,
)


def main():

    # Load first document
    document = next(
        load_jsonl(LEGAL_DOCUMENT_FILE)
    )

    # Semantic splitting
    semantic_chunks = list(
        LegalSplitter().split(document)
    )

    # Length splitting
    length_splitter = LengthSplitter(
        chunk_size=1000,
        overlap=200,
    )

    final_chunks = []

    for chunk in semantic_chunks:
        final_chunks.extend(
            length_splitter.split(chunk)
        )

    print()
    print("=" * 100)
    print(document.title)
    print("=" * 100)

    print()
    print(f"Semantic chunks : {len(semantic_chunks)}")
    print(f"Final chunks    : {len(final_chunks)}")

    for i, chunk in enumerate(final_chunks):

        print()
        print("=" * 100)
        print(f"Chunk {i}")
        print("=" * 100)

        print(f"chunk_id            : {chunk.chunk_id}")
        print(f"document_id         : {chunk.document_id}")
        print(f"article             : {chunk.article}")

        print()

        print(f"title               : {chunk.title}")
        print(f"legal_type          : {chunk.legal_type}")
        print(f"legal_sectors       : {chunk.legal_sectors}")
        print(f"issuing_authority   : {chunk.issuing_authority}")
        print(f"issuance_date       : {chunk.issuance_date}")

        print()

        print(f"chunk_index         : {chunk.chunk_index}")
        print(f"start_char          : {chunk.start_char}")
        print(f"end_char            : {chunk.end_char}")
        print(f"text_length         : {len(chunk.text)}")

        print()
        print("-" * 100)
        print(chunk.text[:700])

        if len(chunk.text) > 700:
            print("\n...")

        print("-" * 100)


if __name__ == "__main__":
    main()