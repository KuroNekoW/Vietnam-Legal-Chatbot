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
    def _safe_str(value) -> str:
        """
        Safely convert any value to string.

        None -> ""
        str  -> stripped string
        other -> str(value)
        """

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def documents(
        metadata_dataset,
        content_dataset,
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

                document_number=DatasetProcessor._safe_str(
                    meta.get("document_number")
                ),

                title=DatasetProcessor._safe_str(
                    meta.get("title")
                ),

                url=DatasetProcessor._safe_str(
                    meta.get("url")
                ),

                legal_type=DatasetProcessor._safe_str(
                    meta.get("legal_type")
                ),

                legal_sectors=DatasetProcessor._safe_str(
                    meta.get("legal_sectors")
                ),

                issuing_authority=DatasetProcessor._safe_str(
                    meta.get("issuing_authority")
                ),

                issuance_date=DatasetProcessor._safe_str(
                    meta.get("issuance_date")
                ),

                signers=DatasetProcessor._safe_str(
                    meta.get("signers")
                ),

                content=DatasetProcessor._safe_str(
                    row.get("content")
                ),
            )