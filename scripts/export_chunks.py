from tqdm import tqdm

from vn_legal_rag.chunking import (
    LegalSplitter,
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


documents = load_jsonl(
    LEGAL_DOCUMENT_FILE
)

semantic_splitter = LegalSplitter()

length_splitter = LengthSplitter(
    chunk_size=1000,
    overlap=200,
)


def generate_chunks():

    for document in tqdm(
        documents,
        total=TOTAL_DOCUMENTS,
        desc="Chunking",
        unit="docs",
        colour="green",
        ncols=100,
    ):

        semantic_chunks = semantic_splitter.split(document)

        for semantic_chunk in semantic_chunks:

            final_chunks = length_splitter.split(
                semantic_chunk
            )

            yield from final_chunks


save_chunks_jsonl(
    generate_chunks(),
    CHUNK_FILE,
)

print("\nChunk export completed!")
print(CHUNK_FILE)