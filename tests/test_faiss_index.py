import numpy as np

from vn_legal_rag.retrieval import (
    FaissIndex,
)


DIM = 1024


def random_vectors(
    n,
):

    x = np.random.randn(
        n,
        DIM,
    ).astype(np.float32)

    x /= np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return x


index = FaissIndex(
    DIM,
)

vectors = random_vectors(
    100
)

index.add(
    vectors
)

print()

print(index)

print()

print("Vectors :", len(index))

query = vectors[10]

scores, ids = index.search(
    query,
    top_k=5,
)

print()

print("Top 5")

for idx, score in zip(
    ids,
    scores,
):

    print(
        idx,
        f"{score:.4f}",
    )