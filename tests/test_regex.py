from vn_legal_rag.chunking.regex import ARTICLE_PATTERN

from vn_legal_rag.config import PROCESSED_DATA_DIR

from vn_legal_rag.utils import load_jsonl


INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "legal_documents.jsonl"
)


documents = load_jsonl(INPUT_FILE)

doc = next(documents)

print("=" * 100)
print("TITLE")
print("=" * 100)
print(doc.title)

print()

parts = ARTICLE_PATTERN.split(doc.content)

parts = [
    p.strip()
    for p in parts
    if p.strip()
]

print(f"Found {len(parts)} articles")

print()

for i, part in enumerate(parts[:5], start=1):

    print("=" * 100)

    print(f"ARTICLE {i}")

    print()

    print(part[:800])

    print()