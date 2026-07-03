from itertools import islice

from vn_legal_rag.ingestion.loader import DatasetLoader


loader = DatasetLoader(
    "vohuutridung/vietnamese-legal-documents",
    "content",
    streaming=True
)

dataset = loader.load()

print(dataset)

sample = next(iter(dataset["data"]))

print()

for k, v in sample.items():
    print(k, type(v))