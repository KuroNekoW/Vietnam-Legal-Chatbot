from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Atomic retrieval unit.
    """

    chunk_id: str

    document_id: int

    article: str

    chunk_index: int

    sub_chunk_index: int

    start_char: int

    end_char: int

    title: str

    legal_type: str

    legal_sectors: str

    issuing_authority: str

    issuance_date: str

    text: str