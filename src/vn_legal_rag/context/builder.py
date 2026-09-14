from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence

from vn_legal_rag.context.models import (
    ArticleContext,
    ContextChunk,
    ContextResult,
    DocumentContext,
)


class ContextBuilder:
    """
    Build structured LLM context from reranked RetrievedChunk objects.

    Responsibilities:
        1. Exact-deduplicate retrieved chunks.
        2. Convert RetrievedChunk -> ContextChunk.
        3. Group by document.
        4. Group by article.
        5. Sort legal hierarchy.
        6. Preserve document metadata.
        7. Render the final structured context.

    This class deliberately does NOT:
        - decide which law is currently effective;
        - compare legal validity across versions;
        - perform fuzzy semantic deduplication;
        - modify the original RetrievedChunk.
    """

    def __init__(
        self,
        *,
        max_documents: int | None = None,
        max_articles_per_document: int | None = None,
        max_chunks_per_article: int | None = None,
    ) -> None:
        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be > 0")

        if (
            max_articles_per_document is not None
            and max_articles_per_document <= 0
        ):
            raise ValueError(
                "max_articles_per_document must be > 0"
            )

        if (
            max_chunks_per_article is not None
            and max_chunks_per_article <= 0
        ):
            raise ValueError(
                "max_chunks_per_article must be > 0"
            )

        self.max_documents = max_documents
        self.max_articles_per_document = max_articles_per_document
        self.max_chunks_per_article = max_chunks_per_article

    def build(
        self,
        chunks: Sequence,
    ) -> ContextResult:
        """
        Build a structured ContextResult from reranked chunks.

        `chunks` is intentionally duck-typed so this builder can work
        directly with the existing RetrievedChunk dataclass without
        introducing a hard circular import.
        """
        input_count = len(chunks)

        if not chunks:
            return ContextResult(
                documents=[],
                input_chunk_count=0,
                deduplicated_chunk_count=0,
                document_count=0,
            )

        deduplicated = self._exact_deduplicate(chunks)

        context_chunks = [
            self._to_context_chunk(chunk)
            for chunk in deduplicated
        ]

        documents = self._build_documents(context_chunks)

        if self.max_documents is not None:
            documents = documents[: self.max_documents]

        return ContextResult(
            documents=documents,
            input_chunk_count=input_count,
            deduplicated_chunk_count=len(context_chunks),
            document_count=len(documents),
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _exact_deduplicate(
        self,
        chunks: Sequence,
    ) -> list:
        """
        Remove exact duplicate chunks.

        Deduplication key:

            document_id
            article
            article_no
            clause
            clause_no
            point
            point_no
            normalized text

        We intentionally do not use score as part of the key.

        If duplicate records exist with different scores, keep the one
        with the highest rerank_score.
        """
        best_by_key: OrderedDict[tuple, object] = OrderedDict()

        for chunk in chunks:
            key = self._dedup_key(chunk)

            existing = best_by_key.get(key)

            if existing is None:
                best_by_key[key] = chunk
                continue

            if self._score(chunk) > self._score(existing):
                best_by_key[key] = chunk

        return list(best_by_key.values())

    @staticmethod
    def _dedup_key(chunk) -> tuple:
        """
        Exact semantic-location key.

        The text is normalized only for whitespace comparison.
        """
        return (
            getattr(chunk, "document_id", None),
            getattr(chunk, "article_no", None),
            ContextBuilder._normalize_label(
                getattr(chunk, "article", None)
            ),
            getattr(chunk, "clause_no", None),
            ContextBuilder._normalize_label(
                getattr(chunk, "clause", None)
            ),
            ContextBuilder._normalize_label(
                getattr(chunk, "point", None)
            ),
            getattr(chunk, "point_no", None),
            ContextBuilder._normalize_text(
                getattr(chunk, "text", "")
            ),
        )

    @staticmethod
    def _score(chunk) -> float:
        score = getattr(chunk, "rerank_score", None)

        if score is None:
            return float("-inf")

        try:
            return float(score)
        except (TypeError, ValueError):
            return float("-inf")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_context_chunk(chunk) -> ContextChunk:
        """
        Convert the existing RetrievedChunk into ContextChunk.
        """
        return ContextChunk(
            chunk_id=getattr(chunk, "chunk_id", ""),
            document_id=getattr(chunk, "document_id", None),

            article=getattr(chunk, "article", None),
            article_no=getattr(chunk, "article_no", None),

            clause=getattr(chunk, "clause", None),
            clause_no=getattr(chunk, "clause_no", None),

            point=getattr(chunk, "point", None),
            point_no=getattr(chunk, "point_no", None),

            chunk_index=getattr(chunk, "chunk_index", None),
            sub_chunk_index=getattr(
                chunk,
                "sub_chunk_index",
                None,
            ),

            title=getattr(chunk, "title", None),
            legal_type=getattr(
                chunk,
                "legal_type",
                None,
            ),
            legal_sectors=getattr(
                chunk,
                "legal_sectors",
                None,
            ),
            issuing_authority=getattr(
                chunk,
                "issuing_authority",
                None,
            ),
            issuance_date=getattr(
                chunk,
                "issuance_date",
                None,
            ),

            url=getattr(chunk, "url", None),
            signers=getattr(chunk, "signers", None),

            text=getattr(chunk, "text", ""),
            rerank_score=getattr(
                chunk,
                "rerank_score",
                None,
            ),
        )

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _build_documents(
        self,
        chunks: Sequence[ContextChunk],
    ) -> list[DocumentContext]:
        """
        Group chunks by document_id.
        """
        groups: OrderedDict[
            int | None,
            list[ContextChunk],
        ] = OrderedDict()

        for chunk in chunks:
            groups.setdefault(
                chunk.document_id,
                [],
            ).append(chunk)

        documents: list[DocumentContext] = []

        for document_id, document_chunks in groups.items():
            document_chunks = sorted(
                document_chunks,
                key=self._chunk_sort_key,
            )

            documents.append(
                self._build_document(
                    document_id=document_id,
                    chunks=document_chunks,
                )
            )

        return documents

    def _build_document(
        self,
        *,
        document_id: int | None,
        chunks: Sequence[ContextChunk],
    ) -> DocumentContext:
        """
        Build one DocumentContext and group its chunks by article.
        """
        first = chunks[0]

        article_groups: OrderedDict[
            tuple,
            list[ContextChunk],
        ] = OrderedDict()

        for chunk in chunks:
            key = self._article_key(chunk)

            article_groups.setdefault(
                key,
                [],
            ).append(chunk)

        articles: list[ArticleContext] = []

        article_items = list(
            article_groups.items()
        )

        article_items.sort(
            key=lambda item: self._article_group_sort_key(
                item[1]
            )
        )

        if self.max_articles_per_document is not None:
            article_items = article_items[
                : self.max_articles_per_document
            ]

        for article_key, article_chunks in article_items:
            article_chunks = sorted(
                article_chunks,
                key=self._chunk_sort_key,
            )

            if self.max_chunks_per_article is not None:
                article_chunks = article_chunks[
                    : self.max_chunks_per_article
                ]

            articles.append(
                ArticleContext(
                    article_key=article_key,
                    article=article_chunks[0].article,
                    article_no=article_chunks[0].article_no,
                    chunks=list(article_chunks),
                )
            )

        return DocumentContext(
            document_id=document_id,

            title=first.title,
            legal_type=first.legal_type,
            legal_sectors=first.legal_sectors,
            issuing_authority=first.issuing_authority,
            issuance_date=first.issuance_date,

            url=first.url,
            signers=first.signers,

            articles=articles,
        )

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    @staticmethod
    def _article_key(
        chunk: ContextChunk,
    ) -> tuple:
        return (
            chunk.document_id,
            chunk.article_no,
            ContextBuilder._natural_sort_key(
                chunk.article
            ),
        )

    @staticmethod
    def _article_group_sort_key(
        chunks: Sequence[ContextChunk],
    ) -> tuple:
        first = chunks[0]

        return (
            ContextBuilder._number_sort_value(
                first.article_no
            ),
            ContextBuilder._natural_sort_key(
                first.article
            ),
        )

    @staticmethod
    def _chunk_sort_key(
        chunk: ContextChunk,
    ) -> tuple:
        return (
            ContextBuilder._number_sort_value(
                chunk.article_no
            ),
            ContextBuilder._natural_sort_key(
                chunk.article
            ),

            ContextBuilder._number_sort_value(
                chunk.clause_no
            ),
            ContextBuilder._natural_sort_key(
                chunk.clause
            ),

            ContextBuilder._point_sort_value(
                chunk.point_no
            ),
            ContextBuilder._natural_sort_key(
                chunk.point
            ),

            ContextBuilder._number_sort_value(
                chunk.chunk_index
            ),
            ContextBuilder._number_sort_value(
                chunk.sub_chunk_index
            ),
        )

    # ------------------------------------------------------------------
    # Sorting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _number_sort_value(
        value: int | str | None,
    ) -> tuple[int, int | str]:
        """
        Sort numeric values before unknown values.

        Examples:
            1 -> (0, 1)
            10 -> (0, 10)
            None -> (1, "")
        """
        if value is None:
            return (1, "")

        if isinstance(value, int):
            return (0, value)

        value_str = str(value).strip()

        if not value_str:
            return (1, "")

        match = re.fullmatch(
            r"-?\d+",
            value_str,
        )

        if match:
            return (0, int(value_str))

        return (1, ContextBuilder._natural_sort_key(value_str))

    @staticmethod
    def _point_sort_value(
        value: int | str | None,
    ) -> tuple[int, object]:
        """
        Sort point numbers such as:

            1
            2
            10
            a
            b
            None

        before arbitrary text.
        """
        if value is None:
            return (2, "")

        value_str = str(value).strip()

        if not value_str:
            return (2, "")

        numeric_match = re.fullmatch(
            r"\d+",
            value_str,
        )

        if numeric_match:
            return (0, int(value_str))

        letter_match = re.fullmatch(
            r"[A-Za-zÀ-ỹ]",
            value_str,
            flags=re.UNICODE,
        )

        if letter_match:
            return (
                1,
                value_str.casefold(),
            )

        return (
            1,
            ContextBuilder._natural_sort_key(
                value_str
            ),
        )

    @staticmethod
    def _natural_sort_key(
        value: str | None,
    ) -> tuple:
        if value is None:
            return ("",)

        text = str(value).strip().casefold()

        parts = re.split(
            r"(\d+)",
            text,
        )

        result: list[object] = []

        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part)

        return tuple(result)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(
        value: str | None,
    ) -> str:
        """
        Normalize whitespace for exact duplicate detection.
        """
        if not value:
            return ""

        return " ".join(
            str(value).split()
        ).strip().casefold()

    @staticmethod
    def _normalize_label(
        value: str | None,
    ) -> str:
        """
        Normalize hierarchy labels while preserving their textual meaning.
        """
        if not value:
            return ""

        return " ".join(
            str(value).split()
        ).strip().casefold()


__all__ = [
    "ContextBuilder",
]