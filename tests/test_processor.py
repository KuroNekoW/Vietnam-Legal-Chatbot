from itertools import islice

from vn_legal_rag.ingestion import (
    DatasetLoader,
    DatasetProcessor
)

from vn_legal_rag.preprocessing import (
    DocumentCleaner,
    MetadataProcessor
)

from vn_legal_rag.models import LegalDocument


DATASET = "vohuutridung/vietnamese-legal-documents"


def test_processor():

    metadata = DatasetLoader(
        DATASET,
        "metadata"
    ).load()

    content = DatasetLoader(
        DATASET,
        "content",
        streaming=True
    ).load()

    documents = DatasetProcessor.documents(
        metadata,
        content
    )

    for doc in islice(documents, 3):

        assert isinstance(
            doc,
            LegalDocument
        )

        doc = MetadataProcessor.process(doc)

        doc = DocumentCleaner.clean(doc)

        print("=" * 80)

        print("ID:", doc.id)

        print()

        print("Title:")

        print(doc.title)

        print()

        print("Number:")

        print(doc.document_number)

        print()

        print("Type:")

        print(doc.legal_type)

        print()

        print("Sector:")

        print(doc.legal_sectors)

        print()

        print("Signer:")

        print(doc.signers)

        print()

        print("Content Preview:")

        print(doc.content[:1000])

        print()

        assert len(doc.title) > 0

        assert len(doc.content) > 0

        assert len(doc.document_number) > 0

        assert isinstance(doc.content, str)


if __name__ == "__main__":

    test_processor()