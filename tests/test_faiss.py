import numpy as np

from vn_legal_rag.retrieval import FaissIndex


def main():

    dim = 1024

    index = FaissIndex(
        dimension=dim,
    )

    embeddings = np.random.rand(
        100,
        dim,
    ).astype(np.float32)

    #
    # normalize
    #

    embeddings /= np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    index.add(
        embeddings
    )

    print()

    print("Dimension :", dim)

    print("Vectors   :", index.ntotal)

    query = embeddings[10]

    scores, ids = index.search(
        query,
        top_k=5,
    )

    print()

    print("Top 5")

    for score, idx in zip(
        scores,
        ids,
    ):

        print(
            idx,
            f"{score:.4f}",
        )


if __name__ == "__main__":

    main()