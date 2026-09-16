from __future__ import annotations

import sys
from pathlib import Path

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


def build_pipeline() -> LegalRAGPipeline:
    print("=" * 80)
    print("INITIALIZING VIETNAMESE LEGAL RAG")
    print("=" * 80)

    # ============================================================
    # EMBEDDING
    # ============================================================

    print("\n[1/8] Loading embedding model...")

    embedding_model = EmbeddingModel()

    print("[OK] Embedding model loaded.")

    # ============================================================
    # QDRANT
    # ============================================================

    print("\n[2/8] Connecting Qdrant...")

    vector_store = QdrantStore(
        collection_name=QDRANT_COLLECTION,
        dimension=embedding_model.dimension,
        database_path=QDRANT_PATH,
    )

    print(
        f"[OK] Qdrant collection: "
        f"{QDRANT_COLLECTION}"
    )

    # ============================================================
    # SQLITE
    # ============================================================

    print("\n[3/8] Opening ChunkStore...")

    chunk_store = ChunkStore(
        database_path=CHUNK_STORE_DB,
    )

    print(
        f"[OK] ChunkStore: "
        f"{CHUNK_STORE_DB}"
    )

    # ============================================================
    # LOCAL LLM
    # ============================================================

    print("\n[4/8] Loading Qwen3-4B...")

    llm = LocalLLM(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CONTEXT_SIZE,
        n_gpu_layers=LLM_GPU_LAYERS,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    print("[OK] Qwen3-4B loaded.")

    # ============================================================
    # QUERY NORMALIZER
    # ============================================================

    print("\n[5/8] Initializing Query Normalizer...")

    query_normalizer = QueryNormalizer(
        llm=llm,
    )

    print("[OK] Query Normalizer ready.")

    # ============================================================
    # RETRIEVER
    # ============================================================

    print("\n[6/8] Initializing Retriever...")

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_store=chunk_store,
        query_normalizer=query_normalizer,
    )

    print("[OK] Retriever ready.")

    # ============================================================
    # RERANKER
    # ============================================================

    print("\n[7/8] Loading Vietnamese Reranker...")

    reranker = Reranker()

    print("[OK] Reranker ready.")

    # ============================================================
    # CONTEXT + GENERATION
    # ============================================================

    print("\n[8/8] Initializing Context + Generator...")

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
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )

    print("[OK] Full RAG pipeline ready.")

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


def print_result(result) -> None:
    print("\n" + "=" * 80)
    print("RAG RESULT")
    print("=" * 80)

    print(
        f"\nOriginal query:\n"
        f"{result.query}"
    )

    if result.normalized_query:
        print(
            f"\nNormalized query:\n"
            f"{result.normalized_query}"
        )

    print(
        f"\nRetrieved candidates: "
        f"{result.retrieved_count}"
    )

    print(
        f"Reranked chunks: "
        f"{result.reranked_count}"
    )

    print(
        f"Selected documents: "
        f"{result.final_context.document_count}"
    )

    print(
        f"Selected chunks: "
        f"{result.final_context.deduplicated_chunk_count}"
    )

    print("\n" + "-" * 80)
    print("FINAL CONTEXT")
    print("-" * 80)

    print(
        result.final_context.rendered_text
    )

    print("\n" + "-" * 80)
    print("FINAL ANSWER")
    print("-" * 80)

    print(result.answer.answer)

    print("\n" + "-" * 80)
    print("CITATIONS")
    print("-" * 80)

    if not result.answer.citations:
        print("(none)")
    else:
        for index, citation in enumerate(
            result.answer.citations,
            start=1,
        ):
            print(f"\n[{index}]")
            print(
                f"Chunk ID      : "
                f"{citation.chunk_id}"
            )
            print(
                f"Document ID   : "
                f"{citation.document_id}"
            )
            print(
                f"Title         : "
                f"{citation.title}"
            )
            print(
                f"Article       : "
                f"{citation.article}"
            )
            print(
                f"Clause        : "
                f"{citation.clause}"
            )
            print(
                f"Point         : "
                f"{citation.point}"
            )
            print(
                f"Issuance date : "
                f"{citation.issuance_date}"
            )

    print("\n" + "-" * 80)
    print(
        "Insufficient evidence: "
        f"{result.answer.insufficient_evidence}"
    )

    if result.answer.note:
        print(
            f"Note: {result.answer.note}"
        )

    print("=" * 80)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            'Usage:\n'
            '  python scripts/run_rag.py '
            '"your legal question"'
        )
        raise SystemExit(1)

    query = " ".join(
        sys.argv[1:]
    )

    pipeline = build_pipeline()

    print("\n")
    print("=" * 80)
    print("RUNNING RAG")
    print("=" * 80)

    result = pipeline.run(
        query,
        retrieval_top_k=30,
        rerank_top_k=10,
    )

    print_result(result)


if __name__ == "__main__":
    main()