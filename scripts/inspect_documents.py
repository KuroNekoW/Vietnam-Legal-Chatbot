from itertools import islice

from vn_legal_rag.ingestion import (
    DatasetLoader,
    DatasetProcessor,
)

DATASET = "vohuutridung/vietnamese-legal-documents"

metadata = DatasetLoader(
    DATASET,
    "metadata"
).load()

content = DatasetLoader(
    DATASET,
    "content",
    streaming=True
).load()

documents = DatasetProcessor.documents(
    metadata,
    content
)

for doc in islice(documents, 3):

    print("=" * 80)

    print(doc.title)

    print()

    print(doc.document_number)

    print()

    print(doc.legal_type)

    print()

    print(doc.content[:1000])

    print()