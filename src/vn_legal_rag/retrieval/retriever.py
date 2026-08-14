from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """
    Một chunk hoàn chỉnh sau khi:

    Qdrant retrieval
        +
    ChunkStore lookup
    """

    chunk_id: str

    score: float

    source_query: str

    document_id: int | None

    article: str | None
    article_no: int | None

    clause: str | None
    clause_no: int | None

    point: str | None
    point_no: str | None

    chunk_index: int | None
    sub_chunk_index: int | None

    start_char: int | None
    end_char: int | None

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
    Retrieval pipeline.

    Input
    -----
    User query

    Pipeline
    --------
    User query
        ↓
    QueryNormalizer
        ↓
    original + normalized query
        ↓
    EmbeddingModel
        ↓
    Qdrant
        ↓
    merge + deduplicate
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
        query_normalizer=None,
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.query_normalizer = query_normalizer

    # ==========================================================
    # Public retrieval
    # ==========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:

        query = query.strip()

        if not query:
            return []

        if candidate_k is None:
            candidate_k = top_k

        if candidate_k < top_k:
            candidate_k = top_k

        # ------------------------------------------------------
        # Build retrieval queries
        # ------------------------------------------------------

        retrieval_queries = [
            (
                query,
                query,
            ),
        ]

        #
        # Nếu có QueryNormalizer:
        # thêm normalized query.
        #

        if self.query_normalizer is not None:

            normalized = (
                self.query_normalizer.normalize(
                    query
                )
            )

            normalized_query = (
                normalized.normalized_query
            )

            #
            # Chỉ thêm nếu thực sự khác query gốc.
            #

            if (
                normalized_query
                and normalized_query.strip().lower()
                != query.lower()
            ):

                retrieval_queries.append(
                    (
                        normalized_query,
                        normalized_query,
                    )
                )

        # ------------------------------------------------------
        # Search Qdrant for each query
        # ------------------------------------------------------

        candidate_map = {}

        for search_query, source_query in retrieval_queries:

            query_vector = (
                self.embedding_model.encode_query(
                    search_query
                )
            )

            results = self.vector_store.search(
                query_vector=query_vector,
                limit=candidate_k,
            )

            for result in results:

                payload = (
                    result.payload
                    or {}
                )

                chunk_id = payload.get(
                    "chunk_id"
                )

                if chunk_id is None:
                    continue

                score = float(
                    result.score
                )

                existing = candidate_map.get(
                    chunk_id
                )

                #
                # Nếu cùng chunk được tìm thấy
                # từ nhiều query, giữ score cao nhất.
                #

                if (
                    existing is None
                    or score > existing["score"]
                ):

                    candidate_map[chunk_id] = {
                        "score": score,
                        "source_query": source_query,
                    }

        if not candidate_map:
            return []

        # ------------------------------------------------------
        # Sort candidates
        # ------------------------------------------------------

        candidates = sorted(
            candidate_map.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        #
        # Chỉ lấy số lượng cần thiết trước khi
        # truy vấn SQLite.
        #

        candidates = candidates[:candidate_k]

        chunk_ids = [
            chunk_id
            for chunk_id, _ in candidates
        ]

        # ------------------------------------------------------
        # ChunkStore lookup
        # ------------------------------------------------------

        records = self.chunk_store.get_many(
            chunk_ids
        )

        # ------------------------------------------------------
        # Build RetrievedChunk
        # ------------------------------------------------------

        retrieved = []

        for chunk_id, info in candidates:

            record = records.get(
                chunk_id
            )

            if record is None:
                continue

            retrieved.append(
                RetrievedChunk(

                    chunk_id=chunk_id,

                    score=info["score"],

                    source_query=info[
                        "source_query"
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

                    start_char=record[
                        "start_char"
                    ],

                    end_char=record[
                        "end_char"
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