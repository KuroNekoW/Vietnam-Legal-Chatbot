from tqdm import tqdm
from collections import Counter

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
    DocumentFilter,
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

    statistics = Counter()

    kept = 0

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

        keep, reason = DocumentFilter.check(doc)

        if not keep:

            statistics[reason] += 1
            continue

        kept += 1

        yield doc

    print()
    print("=" * 60)
    print("FILTER STATISTIC")
    print("=" * 60)

    print(f"Documents kept : {kept:,}")
    print(f"Documents skip : {sum(statistics.values()):,}")
    print()

    if statistics:

        print("Reasons")

        for reason, count in statistics.most_common():

            print(f"{reason:<25} {count:,}")

    print("=" * 60)

save_jsonl(
    preprocess(),
    OUTPUT_FILE,
)

print("\nExport completed!")
print(f"Saved to: {OUTPUT_FILE}")