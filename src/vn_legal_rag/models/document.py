from pydantic import BaseModel, Field


class LegalDocument(BaseModel):
    """
    Standard document model used throughout the project.

    Every module (chunking, embedding, retrieval, LLM...)
    will only interact with this model.
    """

    id: int

    document_number: str = Field(default="")

    title: str = Field(default="")

    url: str = Field(default="")

    legal_type: str = Field(default="")

    legal_sectors: str = Field(default="")

    issuing_authority: str = Field(default="")

    issuance_date: str = Field(default="")

    signers: str = Field(default="")

    content: str = Field(default="")