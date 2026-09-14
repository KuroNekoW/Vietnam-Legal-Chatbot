from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ContextChunk:
    """
    A retrieved chunk prepared for LLM context construction.

    This is intentionally separate from RetrievedChunk so that the
    retrieval layer remains independent from the context-generation layer.
    """

    chunk_id: str
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
    rerank_score: float | None = None

    @property
    def hierarchy(self) -> list[str]:
        """
        Return available legal hierarchy labels in order.
        """
        parts: list[str] = []

        for value in (
            self.article,
            self.clause,
            self.point,
        ):
            if value and value.strip():
                parts.append(value.strip())

        return parts

    @property
    def contextual_text(self) -> str:
        """
        Return the chunk with its legal hierarchy.

        Example:

            Điều 36. Quyền đơn phương...
            Khoản 3
            3. Người lao động...

        The original text is preserved; hierarchy is prepended only for
        context construction.
        """
        parts = self.hierarchy

        if self.text and self.text.strip():
            parts.append(self.text.strip())

        return "\n".join(parts)


@dataclass
class ArticleContext:
    """
    Context for a single article within a legal document.
    """

    article_key: str
    article: str | None
    article_no: int | None

    chunks: list[ContextChunk] = field(default_factory=list)

    @property
    def title(self) -> str | None:
        """
        Prefer the explicit article heading, if available.
        """
        if self.article and self.article.strip():
            return self.article.strip()

        return None

    @property
    def rendered_text(self) -> str:
        """
        Render this article as hierarchical legal context.
        """
        if not self.chunks:
            return ""

        lines: list[str] = []

        if self.article and self.article.strip():
            lines.append(self.article.strip())

        current_clause: str | None = None
        current_point: str | None = None

        for chunk in self.chunks:
            clause_key = chunk.clause

            if clause_key != current_clause:
                current_clause = clause_key
                current_point = None

                if clause_key and clause_key.strip():
                    lines.append(clause_key.strip())

            point_key = chunk.point

            if point_key != current_point:
                current_point = point_key

                if point_key and point_key.strip():
                    lines.append(point_key.strip())

            # Do not duplicate hierarchy already printed above.
            if chunk.text and chunk.text.strip():
                lines.append(chunk.text.strip())

        return "\n".join(lines)


@dataclass
class DocumentContext:
    """
    Context grouped by legal document.
    """

    document_id: int | None

    title: str | None
    legal_type: str | None
    legal_sectors: str | None
    issuing_authority: str | None
    issuance_date: str | None

    url: str | None
    signers: str | None

    articles: list[ArticleContext] = field(default_factory=list)

    @property
    def rendered_text(self) -> str:
        """
        Render the document in a format suitable for an LLM.
        """
        lines: list[str] = []

        lines.append("[VĂN BẢN]")

        if self.title:
            lines.append(f"Tiêu đề: {self.title}")

        if self.legal_type:
            lines.append(f"Loại văn bản: {self.legal_type}")

        if self.issuing_authority:
            lines.append(
                f"Cơ quan ban hành: {self.issuing_authority}"
            )

        if self.issuance_date:
            lines.append(
                f"Ngày ban hành: {self.issuance_date}"
            )

        if self.signers:
            lines.append(
                f"Người ký: {self.signers}"
            )

        for article in self.articles:
            rendered = article.rendered_text.strip()

            if rendered:
                lines.append("")
                lines.append(rendered)

        return "\n".join(lines).strip()


@dataclass
class ContextResult:
    """
    Final structured context returned by ContextBuilder.
    """

    documents: list[DocumentContext] = field(default_factory=list)

    # Number of chunks received from reranker before processing.
    input_chunk_count: int = 0

    # Number of chunks remaining after exact deduplication.
    deduplicated_chunk_count: int = 0

    # Number of source documents represented in the final context.
    document_count: int = 0

    @property
    def rendered_text(self) -> str:
        """
        Render all documents into one LLM-ready context string.
        """
        parts: list[str] = []

        for document in self.documents:
            text = document.rendered_text.strip()

            if text:
                parts.append(text)

        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not bool(self.documents)

    def iter_chunks(self) -> Iterable[ContextChunk]:
        """
        Iterate over all context chunks in final rendering order.
        """
        for document in self.documents:
            for article in document.articles:
                yield from article.chunks