from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """
    Wrapper của FAISS IndexFlatIP.

    Vì embedding đã normalize nên:

        cosine similarity == inner product
    """

    def __init__(
        self,
        dimension: int,
    ):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

    @property
    def ntotal(
        self,
    ) -> int:

        """
        Number of vectors in index.
        """

        return self.index.ntotal

    @property
    def is_empty(
        self,
    ) -> bool:

        return self.ntotal == 0

    def add(
        self,
        vectors: np.ndarray,
    ):

        """
        Add vectors into FAISS.
        """

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim == 1:

            vectors = vectors.reshape(
                1,
                -1,
            )

        if vectors.shape[1] != self.dimension:

            raise ValueError(
                f"Expected dimension={self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        self.index.add(
            vectors
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ):

        """
        Search nearest vectors.

        Returns
        -------
        scores
        indices
        """

        query = np.asarray(
            query,
            dtype=np.float32,
        )

        if query.ndim == 1:

            query = query.reshape(
                1,
                -1,
            )

        scores, indices = self.index.search(
            query,
            top_k,
        )

        return scores[0], indices[0]

    def save(
        self,
        path: str | Path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ):

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        index = faiss.read_index(
            str(path)
        )

        obj = cls(
            index.d
        )

        obj.index = index

        return obj

    def __len__(
        self,
    ):

        return self.ntotal

    def __repr__(
        self,
    ):

        return (
            f"FaissIndex("
            f"dimension={self.dimension}, "
            f"vectors={self.ntotal})"
        )