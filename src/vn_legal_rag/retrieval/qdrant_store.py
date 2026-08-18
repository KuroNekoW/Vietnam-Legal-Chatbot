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

    Responsibilities
    ----------------
    - Connect tới Qdrant server
    - Create collection
    - Add vectors
    - Search vectors
    - Check missing chunks để resume
    - Enable / disable HNSW indexing
    - Count vectors

    Lưu trong Qdrant
    ----------------
    - vector
    - chunk identity
    - legal hierarchy
    - chunk position
    - document metadata

    Không lưu
    ---------
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

        #
        # database_path được giữ lại để tương thích với
        # code/config cũ.
        #
        # Khi chạy Qdrant Docker, tham số này không được sử dụng.
        #

        self.database_path = (
            Path(database_path)
            if database_path is not None
            else None
        )

        self.client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=600,
            prefer_grpc=False,
        )

        #
        # Create collection nếu chưa tồn tại.
        #
        # indexing_threshold = 0:
        # không build HNSW trong lúc import.
        #

        if not self.client.collection_exists(
            collection_name
        ):

            print(
                "Creating Qdrant collection..."
            )

            self.client.create_collection(

                collection_name=collection_name,

                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),

                optimizers_config=(
                    OptimizersConfigDiff(
                        indexing_threshold=0,
                    )
                ),
            )

            print(
                "Collection created."
            )

    # =========================================================
    # Point ID
    # =========================================================

    @staticmethod
    def point_id(
        chunk_id: str,
    ) -> int:
        """
        Chuyển chunk_id thành unsigned 64-bit integer.

        Qdrant chỉ chấp nhận point ID dạng:
        - unsigned integer
        - UUID

        BLAKE2b giúp ID:
        - deterministic
        - ổn định giữa các lần chạy
        - phù hợp với resume
        """

        return int.from_bytes(

            hashlib.blake2b(

                chunk_id.encode(
                    "utf-8"
                ),

                digest_size=8,

            ).digest(),

            "big",

            signed=False,
        )

    # =========================================================
    # Add vectors
    # =========================================================

    def add(
        self,
        embeddings: np.ndarray,
        chunks,
    ):
        """
        Add vectors + metadata vào Qdrant.

        embeddings:
            shape = (N, dimension)

        chunks:
            danh sách Chunk tương ứng.
        """

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings does not "
                "match number of chunks."
            )

        if len(chunks) == 0:
            return

        points = []

        for embedding, chunk in zip(
            embeddings,
            chunks,
        ):

            points.append(

                PointStruct(

                    # ------------------------------------------------
                    # Qdrant point ID
                    # ------------------------------------------------

                    id=self.point_id(
                        chunk.chunk_id
                    ),

                    # ------------------------------------------------
                    # Vector
                    # ------------------------------------------------

                    vector=embedding.tolist(),

                    # ------------------------------------------------
                    # Payload
                    # ------------------------------------------------

                    payload={

                        # --------------------------------------------
                        # Identity
                        # --------------------------------------------

                        "chunk_id": (
                            chunk.chunk_id
                        ),

                        "document_id": (
                            chunk.document_id
                        ),

                        # --------------------------------------------
                        # Legal hierarchy
                        # --------------------------------------------

                        "article": (
                            chunk.article
                        ),

                        "article_no": (
                            chunk.article_no
                        ),

                        "clause": (
                            chunk.clause
                        ),

                        "clause_no": (
                            chunk.clause_no
                        ),

                        "point": (
                            chunk.point
                        ),

                        "point_no": (
                            chunk.point_no
                        ),

                        # --------------------------------------------
                        # Chunk position
                        # --------------------------------------------

                        "chunk_index": (
                            chunk.chunk_index
                        ),

                        "sub_chunk_index": (
                            chunk.sub_chunk_index
                        ),

                        "start_char": (
                            chunk.start_char
                        ),

                        "end_char": (
                            chunk.end_char
                        ),

                        # --------------------------------------------
                        # Document metadata
                        # --------------------------------------------

                        "title": (
                            chunk.title
                        ),

                        "legal_type": (
                            chunk.legal_type
                        ),

                        "legal_sectors": (
                            chunk.legal_sectors
                        ),

                        "issuing_authority": (
                            chunk.issuing_authority
                        ),

                        "issuance_date": (
                            chunk.issuance_date
                        ),
                    },
                )
            )

        #
        # Chia nhỏ request.
        #
        # Điều này tránh gửi request quá lớn lên Qdrant.
        #

        for start in range(
            0,
            len(points),
            self.upsert_batch_size,
        ):

            batch_points = points[
                start:start
                + self.upsert_batch_size
            ]

            self.client.upsert(

                collection_name=(
                    self.collection_name
                ),

                points=batch_points,

                #
                # Chờ Qdrant ghi xong trước khi
                # xử lý tiếp batch.
                #

                wait=True,
            )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query_vector,
        limit: int = 10,
    ):
        """
        Semantic vector search.

        Returns
        -------
        list[ScoredPoint]
        """

        if limit <= 0:
            return []

        result = self.client.query_points(

            collection_name=(
                self.collection_name
            ),

            query=query_vector.tolist(),

            limit=limit,

            #
            # Retrieval cần payload để lấy chunk_id.
            #

            with_payload=True,

            #
            # Không cần lấy lại vector.
            #

            with_vectors=False,
        )

        return result.points

    # =========================================================
    # Resume
    # =========================================================

    def filter_missing(
        self,
        chunks,
    ):
        """
        Trả về các chunk chưa tồn tại trong Qdrant.

        Không load toàn bộ chunk IDs vào RAM.
        Chỉ kiểm tra batch hiện tại.
        """

        if not chunks:
            return []

        ids = [

            self.point_id(
                chunk.chunk_id
            )

            for chunk in chunks
        ]

        records = self.client.retrieve(

            collection_name=(
                self.collection_name
            ),

            ids=ids,

            with_vectors=False,

            with_payload=False,
        )

        existing = {
            point.id
            for point in records
        }

        missing = []

        for chunk in chunks:

            point_id = self.point_id(
                chunk.chunk_id
            )

            if point_id not in existing:

                missing.append(
                    chunk
                )

        return missing

    # =========================================================
    # HNSW indexing
    # =========================================================

    def enable_indexing(
        self,
        indexing_threshold: int = 20_000,
    ):
        """
        Bật lại HNSW indexing sau khi import hoàn tất.
        """

        print(
            "Enabling HNSW indexing..."
        )

        self.client.update_collection(

            collection_name=(
                self.collection_name
            ),

            optimizers_config=(
                OptimizersConfigDiff(
                    indexing_threshold=(
                        indexing_threshold
                    )
                )
            ),
        )

    def disable_indexing(
        self,
    ):
        """
        Tắt HNSW indexing.

        Dùng trong lúc bulk import.
        """

        print(
            "Disabling HNSW indexing..."
        )

        self.client.update_collection(

            collection_name=(
                self.collection_name
            ),

            optimizers_config=(
                OptimizersConfigDiff(
                    indexing_threshold=0,
                )
            ),
        )

    # =========================================================
    # Collection utils
    # =========================================================

    def clear(
        self,
    ):
        """
        Xóa collection hiện tại.
        """

        if self.client.collection_exists(
            self.collection_name
        ):

            print(
                f"Deleting collection: "
                f"{self.collection_name}"
            )

            self.client.delete_collection(
                self.collection_name
            )

    @property
    def ntotal(self) -> int:
        """
        Số point thực tế trong collection.
        """

        return self.client.count(

            collection_name=(
                self.collection_name
            ),

            #
            # exact=True để dùng làm thống kê /
            # điều kiện hoàn thành build.
            #

            exact=True,
        ).count