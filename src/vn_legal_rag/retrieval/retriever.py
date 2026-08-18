from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """
    Một chunk hoàn chỉnh sau khi:

    Qdrant retrieval
        +
    ChunkStore lookup

    Có thể tiếp tục được rerank bằng rerank_score.
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

    rerank_score: float | None = None


class Retriever:
    """
    Semantic retrieval pipeline.

    Pipeline
    --------
    User query
        ↓
    QueryNormalizer
        ↓
    Original query
        +
    normalized query
        +
    keywords
        +
    temporal constraints
        ↓
    E5 query embeddings
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

        retrieval_queries = []

        #
        # Query gốc luôn được giữ lại.
        #

        retrieval_queries.append(
            (
                query,
                "original",
            )
        )

        #
        # Query normalization
        #

        if self.query_normalizer is not None:

            try:

                normalized = (
                    self.query_normalizer.normalize(
                        query
                    )
                )

                normalized_query = (
                    normalized.normalized_query
                )

                #
                # ------------------------------------------------
                # Guardrail
                # ------------------------------------------------
                #
                # Đảm bảo mọi temporal constraint mà LLM
                # nhận diện được vẫn xuất hiện trực tiếp
                # trong query dùng để embedding.
                #

                normalized_query = (
                    self._ensure_constraints(
                        normalized_query,
                        normalized.constraints,
                    )
                )

                normalized_query = (
                    self._ensure_constraints(
                        normalized_query,
                        normalized.temporal_constraints,
                    )
                )

                #
                # ------------------------------------------------
                # Enriched retrieval query
                # ------------------------------------------------
                #
                # normalized_query
                # +
                # keywords
                # +
                # legal_terms
                # +
                # temporal constraints
                #

                enriched_query = (
                    self._build_enriched_query(
                        normalized_query=normalized_query,
                        keywords=normalized.keywords,
                        legal_terms=normalized.legal_terms,
                        constraints=normalized.constraints,
                        temporal_constraints=normalized.temporal_constraints,
                    )
                )

                #
                # Normalized query
                #

                if (
                    normalized_query
                    and normalized_query.strip().lower()
                    != query.lower()
                ):

                    retrieval_queries.append(
                        (
                            normalized_query,
                            "normalized",
                        )
                    )

                #
                # Enriched query
                #
                # Nếu enriched query khác normalized query,
                # thực hiện thêm một search.
                #

                if (
                    enriched_query
                    and enriched_query.strip().lower()
                    != normalized_query.strip().lower()
                ):

                    retrieval_queries.append(
                        (
                            enriched_query,
                            "normalized_enriched",
                        )
                    )

            except Exception as exc:

                #
                # Query normalization không được làm
                # toàn bộ retrieval chết theo.
                #

                print(
                    f"[Retriever] Query normalization failed: "
                    f"{exc}"
                )

                print(
                    "[Retriever] Falling back to original query."
                )

        # ------------------------------------------------------
        # Remove duplicate retrieval queries
        # ------------------------------------------------------

        retrieval_queries = (
            self._deduplicate_queries(
                retrieval_queries
            )
        )

        # ------------------------------------------------------
        # Search Qdrant
        # ------------------------------------------------------

        candidate_map = {}

        for search_query, source_query in retrieval_queries:

            #
            # E5 query embedding
            #

            query_vector = (
                self.embedding_model.encode_query(
                    search_query
                )
            )

            #
            # Qdrant
            #

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
                # Cùng một chunk có thể được tìm thấy
                # bởi original + normalized + enriched.
                #
                # Giữ score cao nhất.
                #

                if (
                    existing is None
                    or score > existing["score"]
                ):

                    candidate_map[chunk_id] = {

                        "score": score,

                        "source_query": (
                            source_query
                        ),

                        "search_query": (
                            search_query
                        ),

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
        # Giới hạn candidate trước khi lookup SQLite.
        #

        candidates = candidates[
            :candidate_k
        ]

        # ------------------------------------------------------
        # ChunkStore lookup
        # ------------------------------------------------------

        chunk_ids = [
            chunk_id
            for chunk_id, _ in candidates
        ]

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

    # ==========================================================
    # Normalized query helpers
    # ==========================================================

    @staticmethod
    def _ensure_constraints(
        text: str,
        constraints: list[str],
    ) -> str:

        if not text:
            return text

        result = text.strip()

        for constraint in constraints:

            constraint = (
                constraint.strip()
            )

            if not constraint:
                continue

            #
            # Case-insensitive containment.
            #

            if (
                constraint.casefold()
                not in result.casefold()
            ):

                result = (
                    f"{result} {constraint}"
                )

        return result

    @staticmethod
    def _build_enriched_query(
        normalized_query: str,
        keywords: list[str],
        legal_terms: list[str],
        constraints: list[str],
        temporal_constraints: list[str],
    ) -> str:
        """
        Tạo query dành riêng cho embedding.

        Không biến thành một đoạn văn quá dài.
        Chỉ bổ sung những tín hiệu retrieval quan trọng.
        """

        parts = [
            normalized_query.strip()
        ]

        #
        # Merge keyword sources nhưng giữ thứ tự.
        #

        retrieval_terms = []

        for term in keywords:

            term = term.strip()

            if (
                term
                and term.casefold()
                not in {
                    x.casefold()
                    for x in retrieval_terms
                }
            ):

                retrieval_terms.append(
                    term
                )

        for term in legal_terms:

            term = term.strip()

            if (
                term
                and term.casefold()
                not in {
                    x.casefold()
                    for x in retrieval_terms
                }
            ):

                retrieval_terms.append(
                    term
                )

        #
        # Constraints
        #

        for constraint in constraints:

            constraint = constraint.strip()

            if (
                constraint
                and constraint.casefold()
                not in {
                    x.casefold()
                    for x in retrieval_terms
                }
            ):

                retrieval_terms.append(
                    constraint
                )

        #
        # Temporal constraints phải được giữ lại.
        #

        for temporal in temporal_constraints:

            temporal = temporal.strip()

            if (
                temporal
                and temporal.casefold()
                not in {
                    x.casefold()
                    for x in retrieval_terms
                }
            ):

                retrieval_terms.append(
                    temporal
                )

        if retrieval_terms:

            parts.append(
                "Từ khóa pháp lý: "
                + "; ".join(
                    retrieval_terms
                )
            )

        return " ".join(
            part
            for part in parts
            if part
        ).strip()

    # ==========================================================
    # Retrieval query helpers
    # ==========================================================

    @staticmethod
    def _deduplicate_queries(
        queries: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:

        seen = set()
        result = []

        for search_query, source_query in queries:

            normalized = (
                search_query
                .strip()
                .casefold()
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(
                (
                    search_query.strip(),
                    source_query,
                )
            )

        return result