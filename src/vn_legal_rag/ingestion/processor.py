from typing import Iterable

from vn_legal_rag.models import LegalDocument


class DatasetProcessor:

    @staticmethod
    def _safe_str(value):

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def documents(
        metadata_dataset,
        content_dataset,
    ) -> Iterable[LegalDocument]:

        metadata_split = list(metadata_dataset.keys())[0]
        content_split = list(content_dataset.keys())[0]

        metadata_iter = iter(metadata_dataset[metadata_split])
        content_iter = iter(content_dataset[content_split])

        for meta, content in zip(metadata_iter, content_iter):

            if meta["id"] != content["id"]:
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
                    content.get("content")
                ),
            )