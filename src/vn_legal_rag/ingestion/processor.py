from typing import Iterable

from vn_legal_rag.models import LegalDocument


class DatasetProcessor:
    """
    Convert HuggingFace dataset records into LegalDocument.
    """

    @staticmethod
    def inspect(dataset):

        print(dataset)

        print()

        split_name = list(dataset.keys())[0]

        sample = dataset[split_name][0]

        print("=" * 60)

        print("SAMPLE")

        print("=" * 60)

        for key, value in sample.items():

            print(f"{key:<20} {type(value)}")

    @staticmethod
    def create_metadata_dict(dataset):

        split_name = list(dataset.keys())[0]

        metadata = {}

        for row in dataset[split_name]:

            metadata[row["id"]] = row

        return metadata

    @staticmethod
    def documents(
        metadata_dataset,
        content_dataset
    ) -> Iterable[LegalDocument]:

        metadata = DatasetProcessor.create_metadata_dict(
            metadata_dataset
        )

        split_name = list(content_dataset.keys())[0]

        for row in content_dataset[split_name]:

            meta = metadata.get(row["id"])

            if meta is None:
                continue

            yield LegalDocument(

                id=meta["id"],

                document_number=meta["document_number"],

                title=meta["title"],

                url=meta["url"],

                legal_type=meta["legal_type"],

                legal_sectors=meta["legal_sectors"],

                issuing_authority=meta["issuing_authority"],

                issuance_date=meta["issuance_date"],

                signers=meta["signers"],

                content=row["content"]
            )