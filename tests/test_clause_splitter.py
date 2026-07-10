from vn_legal_rag.chunking import (
    LegalSplitter,
    ClauseSplitter,
)

from vn_legal_rag.config import LEGAL_DOCUMENT_FILE
from vn_legal_rag.utils import load_jsonl


document = next(
    load_jsonl(
        LEGAL_DOCUMENT_FILE
    )
)

article = list(
    LegalSplitter().split(document)
)[1]

clauses = list(
    ClauseSplitter().split(article)
)

print()

print(article.article)

print()

print(
    "Clauses:",
    len(clauses)
)

for clause in clauses:

    print()

    print("=" * 80)

    print(clause.clause)

    print()

    print(clause.text[:400])