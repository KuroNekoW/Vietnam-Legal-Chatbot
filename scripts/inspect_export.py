from itertools import islice

from vn_legal_rag.config import (
    PROCESSED_DATA_DIR,
)

from vn_legal_rag.utils import (
    load_jsonl,
)


documents = load_jsonl(
    PROCESSED_DATA_DIR /
    "legal_documents.jsonl"
)

for doc in islice(
    documents,
    3,
):

    print("=" * 80)

    print(doc.title)

    print()

    print(doc.document_number)

    print()

    print(doc.content[:500])