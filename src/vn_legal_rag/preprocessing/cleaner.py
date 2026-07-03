import re

from vn_legal_rag.models import LegalDocument


class DocumentCleaner:
    """
    Clean textual fields while preserving the legal document structure.
    """

    @staticmethod
    def clean_text(text: str) -> str:

        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove trailing spaces
        text = "\n".join(
            line.rstrip()
            for line in text.split("\n")
        )

        # Remove duplicate spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @classmethod
    def clean(cls, document: LegalDocument) -> LegalDocument:

        document.title = cls.clean_text(document.title)

        document.content = cls.clean_text(document.content)

        return document