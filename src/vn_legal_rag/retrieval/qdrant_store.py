from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)


class QdrantStore:
    """
    Wrapper của Qdrant.

    Chỉ lưu:

    - vector
    - metadata cần cho retrieval

    Không lưu:

    - text
    - url
    - signers
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        database_path: str | Path | None = None,
        upsert_batch_size: int = 128,
    ):

        self.collection_name = collection_name
        self.upsert_batch_size = upsert_batch_size

        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=600,
            prefer_grpc=False,
        )

        if not self.client.collection_exists(collection_name):

            print("Creating Qdrant collection...")

            self.client.create_collection(
                collection_name=collection_name,

                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),

                #
                # Không build HNSW trong lúc import
                #
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
            )

            print("Collection created.")

    # ----------------------------------------------------
    # Point id
    # ----------------------------------------------------

    @staticmethod
    def point_id(chunk_id: str) -> int:

        return int.from_bytes(
            hashlib.blake2b(
                chunk_id.encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
            signed=False,
        )

    # ----------------------------------------------------
    # Add vectors
    # ----------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        chunks,
    ):

        points = []

        for embedding, chunk in zip(embeddings, chunks):

            points.append(

                PointStruct(

                    id=self.point_id(chunk.chunk_id),

                    vector=embedding.tolist(),

                    payload={

                        "chunk_id": chunk.chunk_id,

                        "document_id": chunk.document_id,

                        "article_no": chunk.article_no,
                        "clause_no": chunk.clause_no,
                        "point_no": chunk.point_no,

                        "title": chunk.title,

                        "legal_type": chunk.legal_type,
                        "legal_sectors": chunk.legal_sectors,
                        "issuing_authority": chunk.issuing_authority,
                        "issuance_date": chunk.issuance_date,

                    },

                )

            )

        #
        # Chia nhỏ request gửi sang Qdrant
        #

        for i in range(
            0,
            len(points),
            self.upsert_batch_size,
        ):

            self.client.upsert(

                collection_name=self.collection_name,

                points=points[
                    i:i + self.upsert_batch_size
                ],

                #
                # Đợi ghi xong mới gửi batch tiếp
                #

                wait=True,

            )

    # ----------------------------------------------------
    # Search
    # ----------------------------------------------------

    def search(
        self,
        query_vector,
        limit: int = 10,
    ):

        return self.client.search(

            collection_name=self.collection_name,

            query_vector=query_vector.tolist(),

            limit=limit,

            with_payload=True,

        )

    # ----------------------------------------------------
    # Resume
    # ----------------------------------------------------

    def filter_missing(
        self,
        chunks,
    ):

        ids = [
            self.point_id(
                chunk.chunk_id
            )
            for chunk in chunks
        ]

        records = self.client.retrieve(

            collection_name=self.collection_name,

            ids=ids,

            with_vectors=False,

            with_payload=False,

        )

        existing = {
            point.id
            for point in records
        }

        result = []

        for chunk in chunks:

            pid = self.point_id(
                chunk.chunk_id
            )

            if pid not in existing:
                result.append(chunk)

        return result

    # ----------------------------------------------------
    # Indexing
    # ----------------------------------------------------

    def enable_indexing(self):

        print("Enabling HNSW indexing...")

        self.client.update_collection(

            collection_name=self.collection_name,

            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20_000,
            ),

        )

    def disable_indexing(self):

        self.client.update_collection(

            collection_name=self.collection_name,

            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,
            ),

        )

    # ----------------------------------------------------
    # Utils
    # ----------------------------------------------------

    def clear(self):

        if self.client.collection_exists(
            self.collection_name
        ):

            self.client.delete_collection(
                self.collection_name
            )

    @property
    def ntotal(self):

        return self.client.count(

            collection_name=self.collection_name,

            exact=True,

        ).count