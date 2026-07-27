from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class QdrantStore:
    """
    Wrapper của Qdrant.

    Chỉ lưu:

    - vector
    - metadata nhỏ

    Không lưu:

    - text
    - url
    - signers
    """

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        database_path: str | Path |None = None,
    ):

        self.collection_name = collection_name

        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=300,
            
        )

        if not self.client.collection_exists(
            collection_name
        ):

            self.client.create_collection(

                collection_name=collection_name,

                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),

            )

    @staticmethod
    def point_id(chunk_id: str) -> int:

        return int.from_bytes(
            hashlib.blake2b(
                chunk_id.encode(),
                digest_size=8
            ).digest(),
            "big"
        )

    def add(
        self,
        embeddings: np.ndarray,
        chunks,
    ):

        points = []

        for embedding, chunk in zip(
            embeddings,
            chunks,
        ):

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

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=False,
        )

    def search(
        self,
        query_vector,
        limit=10,
    ):

        return self.client.search(

            collection_name=self.collection_name,

            query_vector=query_vector.tolist(),

            limit=limit,

            with_payload=True,

        )

    def clear(self):

        if self.client.collection_exists(
            self.collection_name
        ):

            self.client.delete_collection(
                self.collection_name
            )

    def exists(
        self,
        chunk_id: str,
    ) -> bool:

        result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id],
            with_vectors=False,
            with_payload=False,
        )

        return len(result) > 0
    
    def load_existing_chunk_ids(
        self,
    ) -> set[str]:

        ids = set()

        offset = None

        while True:

            points, offset = self.client.scroll(

                collection_name=self.collection_name,

                limit=50000,

                offset=offset,

                with_vectors=False,

                with_payload=False,

            )

            if not points:
                break

            ids.update(
                str(point.id)
                for point in points
            )

            if offset is None:
                break

        return ids

    @property
    def ntotal(self):

        return self.client.count(
            collection_name=self.collection_name,
            exact=False,
        ).count