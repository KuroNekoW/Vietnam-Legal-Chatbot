from vn_legal_rag.embedding import EmbeddingModel


def main():

    model = EmbeddingModel()

    text = """
    Điều 15.
    Nhà nước bảo đảm quyền sử dụng đất...
    """

    vector = model.encode(text)

    print()

    print("Dimension :", len(vector))

    print()

    print("First 10 values:")

    print(vector[:10])

    print()

    print("Norm:")

    print((vector @ vector) ** 0.5)


if __name__ == "__main__":
    main()