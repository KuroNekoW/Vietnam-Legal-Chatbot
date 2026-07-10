from vn_legal_rag.chunking import (
    LegalSplitter,
    ClauseSplitter,
    PointSplitter,
    LengthSplitter,
)

from vn_legal_rag.config import LEGAL_DOCUMENT_FILE

from vn_legal_rag.utils import load_jsonl


N_DOCUMENTS = 500000

legal_splitter = LegalSplitter()
clause_splitter = ClauseSplitter()
point_splitter = PointSplitter()
length_splitter = LengthSplitter(
    chunk_size=1000,
    overlap=200,
)


def main():

    article_total = 0
    clause_total = 0
    point_total = 0
    final_total = 0

    max_article = 0
    max_clause = 0
    max_point = 0
    max_final = 0

    worst_document = None

    for i, document in enumerate(load_jsonl(LEGAL_DOCUMENT_FILE)):

        if i >= N_DOCUMENTS:
            break

        article_count = 0
        clause_count = 0
        point_count = 0
        final_count = 0

        #
        # Điều
        #

        for article in legal_splitter.split(document):

            article_count += 1

            #
            # Khoản
            #

            for clause in clause_splitter.split(article):

                clause_count += 1

                #
                # Điểm
                #

                for point in point_splitter.split(clause):

                    point_count += 1

                    #
                    # Length
                    #

                    for _ in length_splitter.split(point):

                        final_count += 1

        article_total += article_count
        clause_total += clause_count
        point_total += point_count
        final_total += final_count

        max_article = max(max_article, article_count)
        max_clause = max(max_clause, clause_count)
        max_point = max(max_point, point_count)

        if final_count > max_final:

            max_final = final_count
            worst_document = document

    print()

    print("=" * 60)
    print("PIPELINE STATISTIC")
    print("=" * 60)

    print()

    print(f"Documents : {N_DOCUMENTS}")

    print()

    print(f"Average articles : {article_total / N_DOCUMENTS:.2f}")
    print(f"Average clauses  : {clause_total / N_DOCUMENTS:.2f}")
    print(f"Average points   : {point_total / N_DOCUMENTS:.2f}")
    print(f"Average chunks   : {final_total / N_DOCUMENTS:.2f}")

    print()

    print(f"Max articles : {max_article}")
    print(f"Max clauses  : {max_clause}")
    print(f"Max points   : {max_point}")
    print(f"Max chunks   : {max_final}")

    if worst_document is not None:

        print()

        print("Worst document")

        print(f"ID    : {worst_document.id}")
        print(f"Title : {worst_document.title}")


if __name__ == "__main__":
    main()