from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from vn_legal_rag.context.models import (
    ArticleContext,
    ContextChunk,
    ContextResult,
    DocumentContext,
)


@dataclass(frozen=True)
class SpecialCaseRule:
    """
    Describe one legal special-case pattern.

    query_terms:
        Terms that indicate the user explicitly asks about this case.

    chunk_terms:
        Terms that indicate a retrieved chunk belongs to this case.

    penalty:
        Score penalty when the chunk is clearly special-case but
        the user's query does not mention that case.

    boost:
        Score bonus when both query and chunk refer to the same case.
    """

    name: str
    query_terms: tuple[str, ...]
    chunk_terms: tuple[str, ...]
    penalty: float = 0.20
    boost: float = 0.20


@dataclass
class SelectorConfig:
    """
    Configuration for ContextSelector.
    """

    max_documents: int = 3
    max_chunks: int = 6
    max_chunks_per_document: int = 4

    # Main semantic ranking signal.
    rerank_weight: float = 0.70

    # Supporting lexical signal.
    lexical_weight: float = 0.15

    # Small recency signal.
    recency_weight: float = 0.05

    # Special-case awareness.
    special_case_weight: float = 0.10

    min_lexical_overlap: float = 0.05

    # Conservative special-case rules.
    special_case_rules: tuple[
        SpecialCaseRule, ...
    ] = (
        SpecialCaseRule(
            name="domestic_worker",
            query_terms=(
                "người giúp việc gia đình",
                "giúp việc gia đình",
            ),
            chunk_terms=(
                "người giúp việc gia đình",
                "giúp việc gia đình",
            ),
            penalty=0.35,
            boost=0.35,
        ),
        SpecialCaseRule(
            name="special_occupation",
            query_terms=(
                "ngành nghề đặc thù",
                "ngành, nghề đặc thù",
                "công việc đặc thù",
                "ngành nghề công việc đặc thù",
            ),
            chunk_terms=(
                "ngành nghề đặc thù",
                "ngành, nghề đặc thù",
                "công việc đặc thù",
                "một số ngành, nghề, công việc đặc thù",
            ),
            penalty=0.30,
            boost=0.30,
        ),
        SpecialCaseRule(
            name="fixed_term_12_months",
            query_terms=(
                "hợp đồng xác định thời hạn từ 12 tháng",
                "hợp đồng xác định thời hạn 12 tháng",
                "hợp đồng từ 12 tháng trở lên",
            ),
            chunk_terms=(
                "hợp đồng lao động xác định thời hạn từ 12 tháng trở lên",
                "hợp đồng xác định thời hạn từ 12 tháng trở lên",
                "từ 12 tháng trở lên",
            ),
            penalty=0.20,
            boost=0.20,
        ),
    )


class ContextSelector:
    """
    Select useful legal context after ContextBuilder.

    The selector is intentionally heuristic.

    It:
        - uses rerank score as the primary signal;
        - uses lexical overlap as a supporting signal;
        - uses issuance_date as a weak recency signal;
        - detects explicit legal special cases;
        - preserves legal hierarchy.

    It does NOT:
        - determine legal validity;
        - determine effective date;
        - decide which law legally controls another law;
        - replace legal reasoning by a simple "newest wins" rule.
    """

    def __init__(
        self,
        config: SelectorConfig | None = None,
    ) -> None:
        self.config = config or SelectorConfig()

        self._validate_config()

    # ================================================================
    # PUBLIC API
    # ================================================================

    def select(
        self,
        query: str,
        context: ContextResult,
    ) -> ContextResult:
        """
        Select the most useful chunks from an already-built context.
        """
        if context.is_empty:
            return ContextResult(
                documents=[],
                input_chunk_count=context.input_chunk_count,
                deduplicated_chunk_count=0,
                document_count=0,
            )

        query_tokens = self._tokenize(query)

        scored_chunks = self._score_chunks(
            query=query,
            query_tokens=query_tokens,
            chunks=list(context.iter_chunks()),
        )

        if not scored_chunks:
            return ContextResult(
                documents=[],
                input_chunk_count=context.input_chunk_count,
                deduplicated_chunk_count=0,
                document_count=0,
            )

        selected = self._select_chunks(
            scored_chunks
        )

        documents = self._rebuild_documents(
            selected
        )

        return ContextResult(
            documents=documents,
            input_chunk_count=context.input_chunk_count,
            deduplicated_chunk_count=sum(
                len(article.chunks)
                for document in documents
                for article in document.articles
            ),
            document_count=len(documents),
        )

    # ================================================================
    # SCORING
    # ================================================================

    def _score_chunks(
        self,
        *,
        query: str,
        query_tokens: set[str],
        chunks: Sequence[ContextChunk],
    ) -> list[tuple[ContextChunk, float]]:
        """
        Score every chunk.

        Final score:

            rerank
            + lexical
            + recency
            + special-case awareness
        """
        rerank_values = [
            self._safe_score(
                chunk.rerank_score
            )
            for chunk in chunks
            if chunk.rerank_score is not None
        ]

        minimum = (
            min(rerank_values)
            if rerank_values
            else 0.0
        )

        maximum = (
            max(rerank_values)
            if rerank_values
            else 1.0
        )

        scored: list[
            tuple[ContextChunk, float]
        ] = []

        for chunk in chunks:
            rerank_score = (
                self._normalize_rerank_score(
                    self._safe_score(
                        chunk.rerank_score
                    ),
                    minimum,
                    maximum,
                )
            )

            lexical_score = self._lexical_score(
                query_tokens=query_tokens,
                chunk=chunk,
            )

            recency_score = self._recency_score(
                chunk.issuance_date,
                chunks,
            )

            special_case_score = (
                self._special_case_score(
                    query=query,
                    chunk=chunk,
                )
            )

            final_score = (
                self.config.rerank_weight
                * rerank_score
                + self.config.lexical_weight
                * lexical_score
                + self.config.recency_weight
                * recency_score
                + self.config.special_case_weight
                * special_case_score
            )

            scored.append(
                (
                    chunk,
                    final_score,
                )
            )

        scored.sort(
            key=lambda item: (
                item[1],
                self._safe_score(
                    item[0].rerank_score
                ),
                self._article_number(
                    item[0].article_no
                ),
            ),
            reverse=True,
        )

        return scored

    # ================================================================
    # SPECIAL CASE AWARENESS
    # ================================================================

    def _special_case_score(
        self,
        *,
        query: str,
        chunk: ContextChunk,
    ) -> float:
        """
        Return a score in approximately [-1, +1].

        +1:
            query explicitly asks about the special case and
            chunk clearly belongs to it.

         0:
            no special-case relation.

        -1:
            chunk clearly belongs to a special case that the query
            does not mention.

        This is deliberately conservative.
        """
        score = 0.0

        normalized_query = self._normalize_for_matching(
            query
        )

        chunk_text = self._chunk_search_text(
            chunk
        )

        for rule in self.config.special_case_rules:
            query_matches = self._contains_any(
                normalized_query,
                rule.query_terms,
            )

            chunk_matches = self._contains_any(
                chunk_text,
                rule.chunk_terms,
            )

            if not chunk_matches:
                continue

            if query_matches:
                score += rule.boost
            else:
                score -= rule.penalty

        return max(
            -1.0,
            min(1.0, score),
        )

    @staticmethod
    def _chunk_search_text(
        chunk: ContextChunk,
    ) -> str:
        """
        Combine fields that may reveal a special-case condition.
        """
        return ContextSelector._normalize_for_matching(
            " ".join(
                value
                for value in (
                    chunk.article,
                    chunk.clause,
                    chunk.point,
                    chunk.text,
                    chunk.title,
                    chunk.legal_type,
                    chunk.legal_sectors,
                )
                if value
            )
        )

    # ================================================================
    # LEXICAL RELEVANCE
    # ================================================================

    def _lexical_score(
        self,
        *,
        query_tokens: set[str],
        chunk: ContextChunk,
    ) -> float:
        if not query_tokens:
            return 0.0

        searchable_text = self._chunk_search_text(
            chunk
        )

        chunk_tokens = self._tokenize(
            searchable_text
        )

        if not chunk_tokens:
            return 0.0

        overlap = (
            query_tokens
            & chunk_tokens
        )

        ratio = (
            len(overlap)
            / len(query_tokens)
        )

        if ratio < self.config.min_lexical_overlap:
            return 0.0

        return min(
            ratio,
            1.0,
        )

    # ================================================================
    # RECENCY
    # ================================================================

    def _recency_score(
        self,
        issuance_date: str | None,
        chunks: Sequence[ContextChunk],
    ) -> float:
        """
        Weak recency signal.

        issuance_date is NOT treated as effective_date.
        """
        chunk_date = self._parse_date(
            issuance_date
        )

        if chunk_date is None:
            return 0.0

        dates = [
            self._parse_date(
                chunk.issuance_date
            )
            for chunk in chunks
        ]

        dates = [
            value
            for value in dates
            if value is not None
        ]

        if not dates:
            return 0.0

        newest = max(dates)
        oldest = min(dates)

        if newest == oldest:
            return 0.5

        total_days = (
            newest - oldest
        ).days

        if total_days <= 0:
            return 0.5

        relative = (
            chunk_date - oldest
        ).days / total_days

        return max(
            0.0,
            min(1.0, relative),
        )

    # ================================================================
    # CHUNK SELECTION
    # ================================================================

    def _select_chunks(
        self,
        scored_chunks: Sequence[
            tuple[ContextChunk, float]
        ],
    ) -> list[ContextChunk]:
        """
        Select top chunks while keeping document diversity.
        """
        by_document: dict[
            int | None,
            list[
                tuple[ContextChunk, float]
            ],
        ] = defaultdict(list)

        for chunk, score in scored_chunks:
            by_document[
                chunk.document_id
            ].append(
                (chunk, score)
            )

        for items in by_document.values():
            items.sort(
                key=lambda item: item[1],
                reverse=True,
            )

        ranked_documents = sorted(
            by_document.items(),
            key=lambda item: item[1][0][1],
            reverse=True,
        )

        selected_documents = ranked_documents[
            : self.config.max_documents
        ]

        selected: list[ContextChunk] = []

        # First pass:
        # keep the strongest chunk from each selected document.
        for _, items in selected_documents:
            if not items:
                continue

            selected.append(
                items[0][0]
            )

        # Second pass:
        # fill remaining slots.
        for _, items in selected_documents:
            for chunk, _score in items[1:]:
                if (
                    len(selected)
                    >= self.config.max_chunks
                ):
                    break

                if self._same_chunk(
                    chunk,
                    selected,
                ):
                    continue

                document_count = sum(
                    1
                    for existing in selected
                    if existing.document_id
                    == chunk.document_id
                )

                if (
                    document_count
                    >= self.config.max_chunks_per_document
                ):
                    continue

                selected.append(chunk)

            if (
                len(selected)
                >= self.config.max_chunks
            ):
                break

        # Final context order follows legal hierarchy,
        # not selector score.
        selected.sort(
            key=self._legal_sort_key
        )

        return selected

    # ================================================================
    # REBUILD CONTEXT
    # ================================================================

    def _rebuild_documents(
        self,
        chunks: Sequence[ContextChunk],
    ) -> list[DocumentContext]:
        document_map: dict[
            int | None,
            list[ContextChunk],
        ] = defaultdict(list)

        for chunk in chunks:
            document_map[
                chunk.document_id
            ].append(chunk)

        documents: list[
            DocumentContext
        ] = []

        for document_id, document_chunks in (
            document_map.items()
        ):
            document_chunks.sort(
                key=self._legal_sort_key
            )

            first = document_chunks[0]

            article_map: dict[
                tuple,
                list[ContextChunk],
            ] = defaultdict(list)

            for chunk in document_chunks:
                article_map[
                    self._article_key(chunk)
                ].append(chunk)

            articles: list[
                ArticleContext
            ] = []

            for article_key, article_chunks in (
                article_map.items()
            ):
                article_chunks.sort(
                    key=self._legal_sort_key
                )

                first_article_chunk = (
                    article_chunks[0]
                )

                articles.append(
                    ArticleContext(
                        article_key=str(
                            article_key
                        ),
                        article=(
                            first_article_chunk.article
                        ),
                        article_no=(
                            first_article_chunk.article_no
                        ),
                        chunks=article_chunks,
                    )
                )

            articles.sort(
                key=lambda article: (
                    self._number_sort_key(
                        article.article_no
                    ),
                    self._natural_sort_key(
                        article.article
                    ),
                )
            )

            documents.append(
                DocumentContext(
                    document_id=document_id,

                    title=first.title,
                    legal_type=first.legal_type,
                    legal_sectors=first.legal_sectors,
                    issuing_authority=(
                        first.issuing_authority
                    ),
                    issuance_date=(
                        first.issuance_date
                    ),

                    url=first.url,
                    signers=first.signers,

                    articles=articles,
                )
            )

        return documents

    # ================================================================
    # SORTING
    # ================================================================

    @staticmethod
    def _legal_sort_key(
        chunk: ContextChunk,
    ) -> tuple:
        return (
            ContextSelector._number_sort_key(
                chunk.article_no
            ),
            ContextSelector._natural_sort_key(
                chunk.article
            ),
            ContextSelector._number_sort_key(
                chunk.clause_no
            ),
            ContextSelector._natural_sort_key(
                chunk.clause
            ),
            ContextSelector._point_sort_key(
                chunk.point_no
            ),
            ContextSelector._natural_sort_key(
                chunk.point
            ),
            ContextSelector._number_sort_key(
                chunk.chunk_index
            ),
            ContextSelector._number_sort_key(
                chunk.sub_chunk_index
            ),
        )

    @staticmethod
    def _article_key(
        chunk: ContextChunk,
    ) -> tuple:
        return (
            chunk.article_no,
            ContextSelector._natural_sort_key(
                chunk.article
            ),
        )

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _same_chunk(
        chunk: ContextChunk,
        selected: Sequence[ContextChunk],
    ) -> bool:
        return any(
            chunk.chunk_id
            == other.chunk_id
            for other in selected
        )

    @staticmethod
    def _safe_score(
        value: float | None,
    ) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _normalize_rerank_score(
        score: float,
        minimum: float,
        maximum: float,
    ) -> float:
        if maximum == minimum:
            return 1.0

        return (
            score - minimum
        ) / (
            maximum - minimum
        )

    @staticmethod
    def _tokenize(
        text: str | None,
    ) -> set[str]:
        if not text:
            return set()

        text = ContextSelector._normalize_for_matching(
            text
        )

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
            flags=re.UNICODE,
        )

        return {
            token
            for token in text.split()
            if len(token) >= 2
        }

    @staticmethod
    def _normalize_for_matching(
        text: str | None,
    ) -> str:
        if not text:
            return ""

        text = unicodedata.normalize(
            "NFC",
            str(text),
        ).lower()

        text = " ".join(
            text.split()
        )

        return text.strip()

    @staticmethod
    def _contains_any(
        text: str,
        terms: Sequence[str],
    ) -> bool:
        normalized_text = (
            ContextSelector._normalize_for_matching(
                text
            )
        )

        return any(
            ContextSelector._normalize_for_matching(
                term
            ) in normalized_text
            for term in terms
        )

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> date | None:
        if not value:
            return None

        text = str(value).strip()

        formats = (
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%d-%m-%Y",
            "%d/%m/%y",
        )

        for fmt in formats:
            try:
                return datetime.strptime(
                    text,
                    fmt,
                ).date()
            except ValueError:
                continue

        match = re.search(
            r"(\d{1,2})[/-]"
            r"(\d{1,2})[/-]"
            r"(\d{4})",
            text,
        )

        if match:
            try:
                return date(
                    int(match.group(3)),
                    int(match.group(2)),
                    int(match.group(1)),
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _number_sort_key(
        value: int | str | None,
    ) -> tuple:
        if value is None:
            return (1, "")

        if isinstance(value, int):
            return (0, value)

        text = str(value).strip()

        if text.isdigit():
            return (0, int(text))

        return (
            1,
            ContextSelector._natural_sort_key(
                text
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
                result.append(
                    int(part)
                )
            else:
                result.append(part)

        return tuple(result)

    @staticmethod
    def _point_sort_key(
        value: int | str | None,
    ) -> tuple:
        if value is None:
            return (2, "")

        text = str(value).strip()

        if text.isdigit():
            return (
                0,
                int(text),
            )

        if (
            len(text) == 1
            and text.isalpha()
        ):
            return (
                1,
                text.casefold(),
            )

        return (
            1,
            ContextSelector._natural_sort_key(
                text
            ),
        )

    @staticmethod
    def _article_number(
        value: int | str | None,
    ) -> int:
        if value is None:
            return 10**9

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 10**9

    def _validate_config(self) -> None:
        if self.config.max_documents <= 0:
            raise ValueError(
                "max_documents must be > 0"
            )

        if self.config.max_chunks <= 0:
            raise ValueError(
                "max_chunks must be > 0"
            )

        if (
            self.config.max_chunks_per_document
            <= 0
        ):
            raise ValueError(
                "max_chunks_per_document "
                "must be > 0"
            )

        total_weight = (
            self.config.rerank_weight
            + self.config.lexical_weight
            + self.config.recency_weight
            + self.config.special_case_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one selector weight "
                "must be positive."
            )


__all__ = [
    "ContextSelector",
    "SelectorConfig",
    "SpecialCaseRule",
]