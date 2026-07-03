from vn_legal_rag.models import LegalDocument


class MetadataProcessor:
    """
    Normalize metadata fields.
    """

    @staticmethod
    def normalize(value: str) -> str:

        if value is None:
            return ""

        return value.strip()

    @classmethod
    def process(
        cls,
        document: LegalDocument
    ) -> LegalDocument:

        document.document_number = cls.normalize(
            document.document_number
        )

        document.title = cls.normalize(
            document.title
        )

        document.url = cls.normalize(
            document.url
        )

        document.legal_type = cls.normalize(
            document.legal_type
        )

        document.legal_sectors = cls.normalize(
            document.legal_sectors
        )

        document.issuing_authority = cls.normalize(
            document.issuing_authority
        )

        document.issuance_date = cls.normalize(
            document.issuance_date
        )

        document.signers = cls.normalize(
            document.signers
        )

        return document