from vn_legal_rag.chunking import (
    LegalSplitter,
    ClauseSplitter,
    PointSplitter,
)

from vn_legal_rag.config import LEGAL_DOCUMENT_FILE

from vn_legal_rag.utils import load_jsonl


document = next(
    load_jsonl(
        LEGAL_DOCUMENT_FILE
    )
)

articles = list(
    LegalSplitter().split(
        document
    )
)

#
# tìm khoản có điểm
#

for article in articles:

    clauses = ClauseSplitter().split(
        article
    )

    for clause in clauses:

        points = list(
            PointSplitter().split(
                clause
            )
        )

        if len(points) > 1:

            print()

            print(article.article)

            print()

            print(clause.clause)

            print()

            print(
                "Points:",
                len(points)
            )

            for point in points:

                print()

                print("=" * 80)

                print(point.point)

                print()

                print(
                    point.text[:500]
                )

            raise SystemExit