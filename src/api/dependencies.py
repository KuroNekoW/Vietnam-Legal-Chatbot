from __future__ import annotations

from functools import lru_cache

from vn_legal_rag.context import (
    ContextBuilder,
    ContextSelector,
    SelectorConfig,
)
from vn_legal_rag.embedding.model import EmbeddingModel
from vn_legal_rag.generation import AnswerGenerator
from vn_legal_rag.llm.llama_cpp import LocalLLM
from vn_legal_rag.pipeline import LegalRAGPipeline
from vn_legal_rag.query.normalizer import QueryNormalizer
from vn_legal_rag.retrieval import (
    ChunkStore,
    QdrantStore,
    Reranker,
    Retriever,
)

from vn_legal_rag.config import (
    CHUNK_STORE_DB,
    LLM_MODEL_PATH,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)


@lru_cache(maxsize=1)
def get_pipeline() -> LegalRAGPipeline:
    embedding_model = EmbeddingModel()

    vector_store = QdrantStore(
        collection_name=QDRANT_COLLECTION,
        dimension=embedding_model.dimension,
        database_path=QDRANT_PATH,
    )

    chunk_store = ChunkStore(
        database_path=CHUNK_STORE_DB,
    )

    llm = LocalLLM(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CONTEXT_SIZE,
        n_gpu_layers=LLM_GPU_LAYERS,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    query_normalizer = QueryNormalizer(
        llm=llm,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_store=chunk_store,
        query_normalizer=query_normalizer,
    )

    reranker = Reranker()

    context_builder = ContextBuilder()

    context_selector = ContextSelector(
        SelectorConfig(
            max_documents=3,
            max_chunks=6,
            max_chunks_per_document=4,
        )
    )

    answer_generator = AnswerGenerator(
        llm=llm,
        max_tokens=768,
        temperature=0.0,
    )

    return LegalRAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_store=chunk_store,
        query_normalizer=query_normalizer,
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        context_selector=context_selector,
        answer_generator=answer_generator,
    )