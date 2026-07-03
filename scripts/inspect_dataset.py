from vn_legal_rag.ingestion import (
    DatasetLoader,
    DatasetProcessor,
)

DATASET = "vohuutridung/vietnamese-legal-documents"

loader = DatasetLoader(
    DATASET,
    "metadata"
)

dataset = loader.load()

DatasetProcessor.inspect(dataset)