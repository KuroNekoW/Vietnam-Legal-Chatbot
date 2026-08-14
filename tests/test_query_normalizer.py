from vn_legal_rag.config import (
    LLM_MODEL_PATH,
)

from vn_legal_rag.llm import (
    LocalLLM,
)

from vn_legal_rag.query import (
    QueryNormalizer,
)


print()
print("=" * 70)
print("TEST QUERY NORMALIZER")
print("=" * 70)
print()


# ============================================================
# Load LLM
# ============================================================

print("Loading local LLM...")

llm = LocalLLM(
    model_path=LLM_MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    temperature=0.0,
    max_tokens=512,
)

print()


# ============================================================
# Normalizer
# ============================================================

normalizer = QueryNormalizer(
    llm=llm,
)


# ============================================================
# Test queries
# ============================================================

queries = [

    "tôi mua đất giấy tay năm 2015 giờ làm sổ đỏ được không",

    "đất nhà tôi không có sổ thì có bán được không",

    "muốn xin cấp lại căn cước công dân bị mất thì cần giấy tờ gì",

    "hợp đồng lao động ký 2 năm mà công ty cho nghỉ trước hạn thì sao",

]


for query in queries:

    print("-" * 70)

    print(
        f"Original:\n{query}"
    )

    print()

    result = normalizer.normalize(
        query
    )

    print(
        "Normalized:"
    )

    print(
        result.model_dump_json(
            indent=2,
            ensure_ascii=False,
        )
    )

    print()


print("=" * 70)
print("TEST FINISHED")
print("=" * 70)