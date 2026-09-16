from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vn_legal_rag.context import (
    ContextBuilder,
    ContextResult,
    ContextSelector,
)
from vn_legal_rag.embedding.model import EmbeddingModel
from vn_legal_rag.generation import AnswerGenerator, LegalAnswer
from vn_legal_rag.llm.llama_cpp import LocalLLM
from vn_legal_rag.retrieval import (
    ChunkStore,
    QdrantStore,
    Reranker,
    Retriever,
)

# Đổi import này thành path hiện tại của QueryNormalizer
# nếu package query của bạn có tên khác.
from vn_legal_rag.query.normalizer import QueryNormalizer


@dataclass
class RAGPipelineResult:
    """
    Full result of one RAG request.
    """

    query: str
    normalized_query: str | None

    retrieved_count: int
    reranked_count: int

    context_before_selection: ContextResult
    final_context: ContextResult

    answer: LegalAnswer


class LegalRAGPipeline:
    """
    Orchestrates the complete Vietnamese legal RAG pipeline.
    """

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_store: QdrantStore,
        chunk_store: ChunkStore,
        query_normalizer: QueryNormalizer | None,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        context_selector: ContextSelector,
        answer_generator: AnswerGenerator,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_store = chunk_store

        self.query_normalizer = query_normalizer
        self.retriever = retriever
        self.reranker = reranker

        self.context_builder = context_builder
        self.context_selector = context_selector

        self.answer_generator = answer_generator

    def run(
        self,
        query: str,
        *,
        retrieval_top_k: int = 30,
        rerank_top_k: int = 10,
    ) -> RAGPipelineResult:
        """
        Run the complete RAG pipeline.
        """
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        query = query.strip()

        # ============================================================
        # 1. QUERY NORMALIZATION
        # ============================================================

        normalized_query_text: str | None = None

        if self.query_normalizer is not None:
            normalized = self.query_normalizer.normalize(
                query
            )

            normalized_query_text = (
                normalized.normalized_query
            )

        # ============================================================
        # 2. RETRIEVAL
        # ============================================================

        retrieved = self.retriever.retrieve(
            query=query,
            top_k=retrieval_top_k,
        )

        if not retrieved:
            empty_context = ContextResult(
                documents=[],
                input_chunk_count=0,
                deduplicated_chunk_count=0,
                document_count=0,
            )

            answer = self.answer_generator.generate(
                query=query,
                context=empty_context,
            )

            return RAGPipelineResult(
                query=query,
                normalized_query=normalized_query_text,
                retrieved_count=0,
                reranked_count=0,
                context_before_selection=empty_context,
                final_context=empty_context,
                answer=answer,
            )

        # ============================================================
        # 3. RERANK
        # ============================================================

        reranked = self.reranker.rerank(
            query=query,
            chunks=retrieved,
            top_k=rerank_top_k,
        )

        # ============================================================
        # 4. CONTEXT BUILDER
        # ============================================================

        context = self.context_builder.build(
            reranked
        )

        # ============================================================
        # 5. CONTEXT SELECTOR
        # ============================================================

        selected_context = self.context_selector.select(
            query=query,
            context=context,
        )

        # ============================================================
        # 6. ANSWER GENERATION
        # ============================================================

        answer = self.answer_generator.generate(
            query=query,
            context=selected_context,
        )

        return RAGPipelineResult(
            query=query,
            normalized_query=normalized_query_text,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            context_before_selection=context,
            final_context=selected_context,
            answer=answer,
        )