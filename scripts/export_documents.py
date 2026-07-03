from tqdm import tqdm

from vn_legal_rag.config import TOTAL_DOCUMENTS

from vn_legal_rag.config import (
    PROCESSED_DATA_DIR,
)

from vn_legal_rag.ingestion import (
    DatasetLoader,
    DatasetProcessor,
)

from vn_legal_rag.preprocessing import (
    DocumentCleaner,
    MetadataProcessor,
)

from vn_legal_rag.utils import (
    save_jsonl,
)

DATASET = "vohuutridung/vietnamese-legal-documents"

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "legal_documents.jsonl"
)

print("Loading metadata...")

metadata = DatasetLoader(
    DATASET,
    "metadata",
).load()

print("Loading content...")

content = DatasetLoader(
    DATASET,
    "content",
    streaming=True,
).load()

documents = DatasetProcessor.documents(
    metadata,
    content,
)


def preprocess():

    for doc in tqdm(
        documents,
        total=TOTAL_DOCUMENTS,
        desc="Exporting",
        unit="docs",
        colour="green",
        ncols=100,
    ):

        doc = MetadataProcessor.process(doc)
        doc = DocumentCleaner.clean(doc)

        yield doc


save_jsonl(
    preprocess(),
    OUTPUT_FILE,
)

print("\nExport completed!")
print(f"Saved to: {OUTPUT_FILE}")