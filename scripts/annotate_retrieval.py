from __future__ import annotations

import json
from pathlib import Path

from vn_legal_rag.config import (
    CHUNK_STORE_DB,
    LLM_CONTEXT_SIZE,
    LLM_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_TEMPERATURE,
    QDRANT_COLLECTION,
    QDRANT_PATH,
)

from vn_legal_rag.embedding import EmbeddingModel
from vn_legal_rag.llm import LocalLLM
from vn_legal_rag.query import QueryNormalizer

from vn_legal_rag.retrieval import (
    ChunkStore,
    QdrantStore,
    Retriever,
)


# ============================================================
# Config
# ============================================================

EVALUATION_DIR = Path(
    "data/evaluation"
)

EVALUATION_FILE = (
    EVALUATION_DIR
    / "retrieval_eval.jsonl"
)

CANDIDATE_K = 20


# ============================================================
# Evaluation file utilities
# ============================================================

def ensure_directory():
    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_existing_ids() -> set[str]:
    """
    Chỉ load evaluation IDs vào RAM.
    Không load toàn bộ evaluation dataset.
    """

    if not EVALUATION_FILE.exists():
        return set()

    ids = set()

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError:
                continue

            eval_id = record.get(
                "id"
            )

            if eval_id:
                ids.add(eval_id)

    return ids


def get_next_eval_id(
    existing_ids: set[str],
) -> str:

    number = 1

    while True:

        eval_id = (
            f"eval_{number:04d}"
        )

        if eval_id not in existing_ids:
            return eval_id

        number += 1


def save_record(
    record: dict,
):
    """
    Append một evaluation case vào JSONL.
    """

    with EVALUATION_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        file.write("\n")


# ============================================================
# Display helpers
# ============================================================

def print_candidate(
    index: int,
    chunk,
):
    """
    In đầy đủ thông tin cần thiết để annotator
    quyết định relevance.
    """

    print()
    print("=" * 80)
    print(
        f"[{index}]"
    )
    print("=" * 80)

    print(
        f"Qdrant score       : "
        f"{chunk.score:.6f}"
    )

    print(
        f"Source query       : "
        f"{chunk.source_query}"
    )

    print(
        f"Chunk ID           : "
        f"{chunk.chunk_id}"
    )

    print(
        f"Document ID        : "
        f"{chunk.document_id}"
    )

    print(
        f"Title              : "
        f"{chunk.title}"
    )

    print(
        f"Legal type         : "
        f"{chunk.legal_type}"
    )

    print(
        f"Issuing authority  : "
        f"{chunk.issuing_authority}"
    )

    print(
        f"Issuance date      : "
        f"{chunk.issuance_date}"
    )

    print()

    print(
        f"Article no         : "
        f"{chunk.article_no}"
    )

    print(
        f"Article            : "
        f"{chunk.article}"
    )

    print(
        f"Clause no          : "
        f"{chunk.clause_no}"
    )

    print(
        f"Clause             : "
        f"{chunk.clause}"
    )

    print(
        f"Point no           : "
        f"{chunk.point_no}"
    )

    print(
        f"Point              : "
        f"{chunk.point}"
    )

    print(
        f"Chunk index        : "
        f"{chunk.chunk_index}"
    )

    print(
        f"Sub chunk index    : "
        f"{chunk.sub_chunk_index}"
    )

    print()

    print("Text:")
    print(
        chunk.text
    )


def print_candidates(
    candidates,
):
    """
    Hiển thị candidate list từ Retriever.
    """

    print()
    print("=" * 80)
    print("RETRIEVAL CANDIDATES")
    print("=" * 80)

    print(
        f"Candidates: {len(candidates)}"
    )

    for index, chunk in enumerate(
        candidates,
        start=1,
    ):

        print_candidate(
            index,
            chunk,
        )

        print()


# ============================================================
# Annotation parser
# ============================================================

def parse_annotations(
    user_input: str,
    candidates,
):
    """
    Parse:

        1:3 4:2 7:1

    thành:

        candidate #1 -> relevance 3
        candidate #4 -> relevance 2
        candidate #7 -> relevance 1
    """

    user_input = user_input.strip()

    if not user_input:
        return []

    annotations = []

    seen_indexes = set()

    for token in user_input.split():

        if ":" not in token:
            continue

        index_text, score_text = token.split(
            ":",
            1,
        )

        try:

            index = int(
                index_text
            )

            relevance = int(
                score_text
            )

        except ValueError:

            continue

        if index < 1:
            continue

        if index > len(candidates):
            continue

        if relevance not in {
            1,
            2,
            3,
        }:
            continue

        if index in seen_indexes:
            continue

        seen_indexes.add(
            index
        )

        chunk = candidates[
            index - 1
        ]

        annotations.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "article_no": chunk.article_no,
                "clause_no": chunk.clause_no,
                "point_no": chunk.point_no,
                "relevance": relevance,
            }
        )

    return annotations


# ============================================================
# Main
# ============================================================

def main():

    ensure_directory()

    existing_ids = (
        load_existing_ids()
    )

    print()
    print("=" * 80)
    print("RETRIEVAL EVALUATION ANNOTATOR")
    print("=" * 80)
    print()

    print(
        f"Evaluation file : "
        f"{EVALUATION_FILE}"
    )

    print(
        f"Existing cases  : "
        f"{len(existing_ids):,}"
    )

    print()

    # ========================================================
    # Load local LLM
    # ========================================================

    print(
        "Loading local Query Normalizer..."
    )

    llm = LocalLLM(

        model_path=LLM_MODEL_PATH,

        n_ctx=LLM_CONTEXT_SIZE,

        n_gpu_layers=LLM_GPU_LAYERS,

        temperature=LLM_TEMPERATURE,

        max_tokens=LLM_MAX_TOKENS,
    )

    normalizer = QueryNormalizer(
        llm
    )

    print()

    # ========================================================
    # Embedding model
    # ========================================================

    print(
        "Loading embedding model..."
    )

    embedding_model = EmbeddingModel()

    print()

    # ========================================================
    # Qdrant
    # ========================================================

    print(
        "Connecting Qdrant..."
    )

    qdrant = QdrantStore(

        collection_name=QDRANT_COLLECTION,

        dimension=embedding_model.dimension,

        database_path=QDRANT_PATH,
    )

    print(
        f"Qdrant vectors : "
        f"{qdrant.ntotal:,}"
    )

    print()

    # ========================================================
    # Chunk store
    # ========================================================

    print(
        "Opening ChunkStore..."
    )

    chunk_store = ChunkStore(
        CHUNK_STORE_DB
    )

    print(
        f"Stored chunks : "
        f"{chunk_store.count:,}"
    )

    print()

    # ========================================================
    # Retriever
    # ========================================================

    retriever = Retriever(

        embedding_model=embedding_model,

        vector_store=qdrant,

        chunk_store=chunk_store,

        query_normalizer=normalizer,
    )

    # ========================================================
    # Interactive annotation loop
    # ========================================================

    try:

        while True:

            print()
            print("-" * 80)
            print(
                "Enter an evaluation query."
            )
            print(
                "Commands: q = quit"
            )
            print("-" * 80)

            query = input(
                "Query: "
            ).strip()

            if query.lower() == "q":
                break

            if not query:
                continue

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            print()
            print(
                "Running Query Normalizer..."
            )

            normalized = (
                normalizer.normalize(
                    query
                )
            )

            print()
            print(
                "Original query:"
            )

            print(
                query
            )

            print()
            print(
                "Normalized query:"
            )

            print(
                normalized.normalized_query
            )

            print()
            print(
                "Question intent:"
            )

            print(
                normalized.question_intent
            )

            print()
            print(
                "Keywords:"
            )

            for keyword in (
                normalized.keywords
            ):

                print(
                    f"  - {keyword}"
                )

            print()
            print(
                "Constraints:"
            )

            for constraint in (
                normalized.constraints
            ):

                print(
                    f"  - {constraint}"
                )

            print()
            print(
                "Temporal constraints:"
            )

            for temporal in (
                normalized.temporal_constraints
            ):

                print(
                    f"  - {temporal}"
                )

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            print()
            print(
                "Running retrieval..."
            )

            candidates = (
                retriever.retrieve(
                    query=query,
                    top_k=CANDIDATE_K,
                    candidate_k=CANDIDATE_K,
                )
            )

            if not candidates:

                print()
                print(
                    "No candidates found."
                )

                continue

            # ------------------------------------------------
            # Show candidates
            # ------------------------------------------------

            print_candidates(
                candidates
            )

            # ------------------------------------------------
            # Annotation
            # ------------------------------------------------

            print()
            print("=" * 80)
            print("ANNOTATION")
            print("=" * 80)
            print()

            print(
                "Relevance levels:"
            )

            print(
                "  3 = directly answers the question"
            )

            print(
                "  2 = highly relevant / needed"
            )

            print(
                "  1 = related but not necessary"
            )

            print(
                "  0 = not relevant / do not select"
            )

            print()

            print(
                "Format:"
            )

            print(
                "  1:3 4:2 7:1"
            )

            print()

            annotation_input = input(
                "Relevant candidates: "
            ).strip()

            annotations = (
                parse_annotations(
                    annotation_input,
                    candidates,
                )
            )

            # ------------------------------------------------
            # Optional metadata
            # ------------------------------------------------

            print()

            category = input(
                "Category: "
            ).strip()

            difficulty = input(
                "Difficulty "
                "[easy/medium/hard]: "
            ).strip().lower()

            if difficulty not in {
                "easy",
                "medium",
                "hard",
            }:

                difficulty = "medium"

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            eval_id = get_next_eval_id(
                existing_ids
            )

            record = {

                "id": eval_id,

                "query": query,

                "category": category,

                "difficulty": difficulty,

                "relevant_chunks": annotations,

            }

            save_record(
                record
            )

            existing_ids.add(
                eval_id
            )

            print()
            print("=" * 80)
            print("SAVED")
            print("=" * 80)

            print(
                f"Evaluation ID : "
                f"{eval_id}"
            )

            print(
                f"Relevant chunks : "
                f"{len(annotations)}"
            )

            print(
                f"File : "
                f"{EVALUATION_FILE}"
            )

            print()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Annotation interrupted."
        )

    finally:

        chunk_store.close()

    print()
    print("=" * 80)
    print("ANNOTATION FINISHED")
    print("=" * 80)
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()