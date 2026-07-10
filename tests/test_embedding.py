import numpy as np

from vn_legal_rag.embedding import EmbeddingModel


def test_single_encode(model: EmbeddingModel):

    print("=" * 80)
    print("SINGLE ENCODE")
    print("=" * 80)

    text = """
    Điều 15. Nhà nước bảo đảm quyền sử dụng đất của người dân.
    """

    vector = model.encode(text)

    print(f"Dimension : {len(vector)}")
    print()

    print("First 10 values:")
    print(vector[:10])

    print()

    print("Norm:")
    print(np.linalg.norm(vector))


def test_batch_encode(model: EmbeddingModel):

    print()
    print("=" * 80)
    print("BATCH ENCODE")
    print("=" * 80)

    texts = [
        "Luật Đất đai năm 2024.",
        "Luật Doanh nghiệp.",
        "Bộ luật Hình sự.",
        "Luật Giao thông đường bộ.",
    ]

    vectors = model.encode_batch(texts)

    print(f"Shape : {vectors.shape}")
    print()

    for i, vector in enumerate(vectors):

        print(
            f"Vector {i}: "
            f"dimension={len(vector)}, "
            f"norm={np.linalg.norm(vector):.4f}"
        )


def test_similarity(model: EmbeddingModel):

    print()
    print("=" * 80)
    print("SIMILARITY")
    print("=" * 80)

    query = "quyền sử dụng đất"

    documents = [
        "Luật đất đai quy định quyền sử dụng đất của người dân.",
        "Thuế thu nhập doanh nghiệp áp dụng cho công ty.",
        "Người sử dụng đất được cấp giấy chứng nhận quyền sử dụng đất.",
        "Quy định xử phạt vi phạm giao thông đường bộ.",
    ]

    q = model.encode(query)

    embeddings = model.encode_batch(documents)

    scores = embeddings @ q

    ranking = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Query: {query}")
    print()

    for i, (doc, score) in enumerate(ranking, 1):

        print(f"{i}. score = {score:.4f}")
        print(doc)
        print()


def main():

    print()
    print("Loading embedding model...")
    print()

    model = EmbeddingModel()

    test_single_encode(model)

    test_batch_encode(model)

    test_similarity(model)


if __name__ == "__main__":
    main()