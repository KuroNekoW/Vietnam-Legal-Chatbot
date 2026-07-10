from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """
    Wrapper around FAISS IndexFlatIP.

    Responsibilities
    ----------------
    - create index
    - add embeddings
    - search
    - save
    - load
    """

    def __init__(
        self,
        dimension: int,
    ):

        self.dimension = dimension

        #
        # Cosine similarity
        # (vectors are already normalized)
        #

        self.index = faiss.IndexFlatIP(
            dimension
        )

    @property
    def ntotal(
        self,
    ) -> int:

        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
    ) -> None:

        embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

        self.index.add(
            embeddings
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ):

        if query_embedding.ndim == 1:

            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        scores, indices = self.index.search(
            query_embedding.astype(
                np.float32,
                copy=False,
            ),
            top_k,
        )

        return scores[0], indices[0]

    def save(
        self,
        path: Path,
    ):

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
        path: Path,
    ):

        index = faiss.read_index(
            str(path)
        )

        obj = cls(
            index.d
        )

        obj.index = index

        return obj