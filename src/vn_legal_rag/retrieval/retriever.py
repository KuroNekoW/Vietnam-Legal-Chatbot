from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """
    Một chunk được retrieve từ Qdrant + ChunkStore.
    """

    chunk_id: str

    score: float

    document_id: int | None

    article: str | None
    article_no: int | None

    clause: str | None
    clause_no: int | None

    point: str | None
    point_no: str | None

    chunk_index: int | None
    sub_chunk_index: int | None

    title: str | None

    legal_type: str | None
    legal_sectors: str | None

    issuing_authority: str | None
    issuance_date: str | None

    url: str | None
    signers: str | None

    text: str


class Retriever:
    """
    Semantic retrieval pipeline.

    Question
        ↓
    EmbeddingModel
        ↓
    Qdrant
        ↓
    chunk_id
        ↓
    ChunkStore
        ↓
    RetrievedChunk
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        chunk_store,
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_store = chunk_store

    # ==========================================================
    # Retrieve
    # ==========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:

        if not query or not query.strip():
            return []

        query = query.strip()

        # ------------------------------------------------------
        # Query embedding
        # ------------------------------------------------------

        query_vector = (
            self.embedding_model.encode_query(
                query
            )
        )

        # ------------------------------------------------------
        # Qdrant search
        # ------------------------------------------------------

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
        )

        if not results:
            return []

        # ------------------------------------------------------
        # Extract chunk IDs
        # ------------------------------------------------------

        chunk_ids = []

        scores = {}

        for result in results:

            payload = result.payload or {}

            chunk_id = payload.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            chunk_ids.append(
                chunk_id
            )

            scores[chunk_id] = float(
                result.score
            )

        if not chunk_ids:
            return []

        # ------------------------------------------------------
        # Retrieve complete chunk records
        # ------------------------------------------------------

        records = self.chunk_store.get_many(
            chunk_ids
        )

        # ------------------------------------------------------
        # Merge Qdrant score + chunk record
        # ------------------------------------------------------

        retrieved = []

        for chunk_id in chunk_ids:

            record = records.get(
                chunk_id
            )

            if record is None:
                continue

            retrieved.append(
                RetrievedChunk(

                    chunk_id=chunk_id,

                    score=scores[
                        chunk_id
                    ],

                    document_id=record[
                        "document_id"
                    ],

                    article=record[
                        "article"
                    ],

                    article_no=record[
                        "article_no"
                    ],

                    clause=record[
                        "clause"
                    ],

                    clause_no=record[
                        "clause_no"
                    ],

                    point=record[
                        "point"
                    ],

                    point_no=record[
                        "point_no"
                    ],

                    chunk_index=record[
                        "chunk_index"
                    ],

                    sub_chunk_index=record[
                        "sub_chunk_index"
                    ],

                    title=record[
                        "title"
                    ],

                    legal_type=record[
                        "legal_type"
                    ],

                    legal_sectors=record[
                        "legal_sectors"
                    ],

                    issuing_authority=record[
                        "issuing_authority"
                    ],

                    issuance_date=record[
                        "issuance_date"
                    ],

                    url=record[
                        "url"
                    ],

                    signers=record[
                        "signers"
                    ],

                    text=record[
                        "text"
                    ],
                )
            )

        return retrieved