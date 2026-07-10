from tqdm import tqdm

from vn_legal_rag.chunking import (
    LegalSplitter,
    ClauseSplitter,
    PointSplitter,
    LengthSplitter,
)

from vn_legal_rag.config import (
    LEGAL_DOCUMENT_FILE,
    CHUNK_FILE,
    TOTAL_DOCUMENTS,
)

from vn_legal_rag.utils import (
    load_jsonl,
    save_chunks_jsonl,
)


semantic_splitter = LegalSplitter()
clause_splitter = ClauseSplitter()
point_splitter = PointSplitter()

length_splitter = LengthSplitter(
    chunk_size=1000,
    overlap=200,
)


def generate_chunks():

    total_chunks = 0

    documents = load_jsonl(
        LEGAL_DOCUMENT_FILE,
    )

    progress = tqdm(
        documents,
        total=TOTAL_DOCUMENTS,
        desc="Chunking",
        unit="docs",
        colour="green",
        dynamic_ncols=True,
    )

    for document in progress:

        #
        # Điều
        #

        for article in semantic_splitter.split(document):

            #
            # Khoản
            #

            for clause in clause_splitter.split(article):

                #
                # Điểm
                #

                for point in point_splitter.split(clause):

                    #
                    # Length split
                    #

                    for chunk in length_splitter.split(point):

                        total_chunks += 1

                        yield chunk

    print()
    print(f"Total chunks: {total_chunks:,}")


if __name__ == "__main__":

    save_chunks_jsonl(
        generate_chunks(),
        CHUNK_FILE,
    )

    print()
    print("=" * 60)
    print("Chunk export completed")
    print("=" * 60)
    print(CHUNK_FILE)
    print()