from vn_legal_rag.ingestion import DatasetLoader

def test_loader():

    loader = DatasetLoader(
        "vohuutridung/vietnamese-legal-documents",
        "metadata"
    )

    dataset = loader.load()

    assert dataset is not None

    print()

    print("Dataset loaded successfully.")

    print(dataset)


if __name__ == "__main__":

    test_loader()