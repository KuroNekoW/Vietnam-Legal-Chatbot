from __future__ import annotations

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

from vn_legal_rag.llm import LocalLLM

from vn_legal_rag.query import QueryNormalizer

from vn_legal_rag.context import ContextBuilder

from vn_legal_rag.retrieval import (
    ChunkStore,
    QdrantStore,
    Reranker,
    Retriever,
)

from vn_legal_rag.embedding.model import EmbeddingModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUERY = (
    "Người lao động đơn phương chấm dứt hợp đồng lao động "
    "thì phải báo trước bao nhiêu ngày?"
)

TOP_K_RETRIEVAL = 30
TOP_K_RERANK = 10


# ============================================================
# HELPERS
# ============================================================

def print_separator(title: str = "") -> None:
    print("\n" + "=" * 80)

    if title:
        print(title)
        print("=" * 80)


def print_retrieved_results(results, top_n: int = 10) -> None:
    print_separator("RERANKED RESULTS")

    for index, result in enumerate(results[:top_n], start=1):
        print(f"\n[{index}]")
        print(f"Chunk ID       : {result.chunk_id}")
        print(f"Document ID    : {result.document_id}")
        print(f"Article No     : {result.article_no}")
        print(f"Article        : {result.article}")
        print(f"Clause No      : {result.clause_no}")
        print(f"Clause         : {result.clause}")
        print(f"Point No       : {result.point_no}")
        print(f"Point          : {result.point}")
        print(f"Title          : {result.title}")
        print(f"Legal type     : {result.legal_type}")
        print(f"Issuing agency : {result.issuing_authority}")
        print(f"Issuance date  : {result.issuance_date}")
        print(f"Qdrant score   : {result.score}")
        print(f"Rerank score   : {result.rerank_score}")
        print(f"Text           : {result.text}")


def print_context(context_result) -> None:
    print_separator("FINAL CONTEXT")

    print(
        f"Input chunks          : "
        f"{context_result.input_chunk_count}"
    )

    print(
        f"After exact dedup     : "
        f"{context_result.deduplicated_chunk_count}"
    )

    print(
        f"Documents             : "
        f"{context_result.document_count}"
    )

    print()

    print(context_result.rendered_text)


# ============================================================
# MAIN INTEGRATION TEST
# ============================================================

def main() -> None:
    print_separator("CONTEXT INTEGRATION TEST")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Query        : {QUERY}")

    # ============================================================
    # Local LLM
    # ============================================================

    print("Loading local LLM...")

    llm = LocalLLM(

        model_path=LLM_MODEL_PATH,

        n_ctx=LLM_CONTEXT_SIZE,

        n_gpu_layers=LLM_GPU_LAYERS,

        temperature=LLM_TEMPERATURE,

        max_tokens=LLM_MAX_TOKENS,

    )

    normalizer = QueryNormalizer(
        llm=llm
    )

    print()

    # ============================================================
    # Normalize query
    # ============================================================

    print("-" * 70)
    print("QUERY NORMALIZATION")
    print("-" * 70)

    print()
    print("Original query:")
    print(QUERY)

    normalized = normalizer.normalize(
        QUERY
    )

    normalized_query = (
        normalized.normalized_query
    )

    print()
    print("Normalized query:")
    print(normalized_query)

    print()
    print("Question intent:")

    print(
        f"  {normalized.question_intent}"
    )

    print()

    print("Keywords:")

    for keyword in normalized.keywords:

        print(
            f"  - {keyword}"
        )

    print()

    print("Constraints:")

    for constraint in normalized.constraints:

        print(
            f"  - {constraint}"
        )

    print()

    print("Temporal constraints:")

    for temporal in normalized.temporal_constraints:

        print(
            f"  - {temporal}"
        )

    print()

    # --------------------------------------------------------
    # 1. Load embedding model
    # --------------------------------------------------------

    print_separator("1. LOAD EMBEDDING MODEL")

    embedding_model = EmbeddingModel()

    print("Embedding model loaded successfully.")

    # --------------------------------------------------------
    # 2. Connect Qdrant
    # --------------------------------------------------------

    print_separator("2. CONNECT QDRANT")

    vector_store = QdrantStore(
        collection_name=QDRANT_COLLECTION,
        dimension=embedding_model.dimension,
        database_path=QDRANT_PATH,
    )

    print("Qdrant connected successfully.")

    # --------------------------------------------------------
    # 3. Load SQLite ChunkStore
    # --------------------------------------------------------

    print_separator("3. LOAD CHUNK STORE")

    chunk_store = ChunkStore(
        CHUNK_STORE_DB
    )

    print("ChunkStore loaded successfully.")

    # --------------------------------------------------------
    # 4. Initialize Retriever
    # --------------------------------------------------------

    print_separator("4. INITIALIZE RETRIEVER")

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_store=chunk_store,
        query_normalizer=normalizer,
    )

    print("Retriever initialized successfully.")

    # --------------------------------------------------------
    # 5. Retrieve candidates
    # --------------------------------------------------------

    print_separator("5. RETRIEVAL")

    print("Running retrieval...")

    retrieved = retriever.retrieve(
        query=QUERY,
        top_k=TOP_K_RETRIEVAL,
    )

    print(f"Retrieved candidates: {len(retrieved)}")

    if not retrieved:
        raise RuntimeError(
            "Retriever returned no results."
        )

    # --------------------------------------------------------
    # 6. Initialize reranker
    # --------------------------------------------------------

    print_separator("6. INITIALIZE RERANKER")

    reranker = Reranker()

    print("Reranker initialized successfully.")

    # --------------------------------------------------------
    # 7. Rerank
    # --------------------------------------------------------

    print_separator("7. RERANKING")

    print("Running reranker...")

    reranked = reranker.rerank(
        query=QUERY,
        chunks=retrieved,
        top_k=TOP_K_RERANK,
    )

    print(f"Reranked results: {len(reranked)}")

    if not reranked:
        raise RuntimeError(
            "Reranker returned no results."
        )

    print_retrieved_results(
        reranked,
        top_n=TOP_K_RERANK,
    )

    # --------------------------------------------------------
    # 8. Context Builder
    # --------------------------------------------------------

    print_separator("8. CONTEXT BUILDER")

    context_builder = ContextBuilder()

    print("Building structured context...")

    context_result = context_builder.build(
        reranked
    )

    if context_result.is_empty:
        raise RuntimeError(
            "ContextBuilder returned empty context."
        )

    from vn_legal_rag.context import ContextSelector

    selector = ContextSelector()

    selected_context = selector.select(
        query=QUERY,
        context=context_result,
    )

    print_context(selected_context)

    # --------------------------------------------------------
    # 9. Basic sanity checks
    # --------------------------------------------------------

    print_separator("9. SANITY CHECKS")

    assert selected_context.input_chunk_count == len(
        reranked
    )

    assert (
        selected_context.deduplicated_chunk_count
        <= selected_context.input_chunk_count
    )

    assert selected_context.document_count > 0

    rendered = selected_context.rendered_text.strip()

    assert rendered

    print(
        "[PASS] Input chunk count is correct."
    )

    print(
        "[PASS] Deduplication count is valid."
    )

    print(
        "[PASS] At least one document exists."
    )

    print(
        "[PASS] Rendered context is not empty."
    )

    # --------------------------------------------------------
    # 10. Legal-content sanity checks
    # --------------------------------------------------------

    print_separator("10. LEGAL CONTENT CHECKS")

    lower_context = rendered.casefold()

    keywords = [
        "đơn phương",
        "chấm dứt",
        "hợp đồng lao động",
        "báo trước",
    ]

    for keyword in keywords:
        if keyword.casefold() in lower_context:
            print(
                f"[PASS] Context contains: {keyword}"
            )
        else:
            print(
                f"[WARN] Context does not contain: "
                f"{keyword}"
            )

    # --------------------------------------------------------
    # 11. Final result
    # --------------------------------------------------------

    print_separator("RESULT")

    print("Context integration test completed.")
    print(
        "Pipeline: "
        "Retriever -> Reranker -> ContextBuilder"
    )


if __name__ == "__main__":
    main()