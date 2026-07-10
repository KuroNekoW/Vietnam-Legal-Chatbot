from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Atomic retrieval unit.
    """

    chunk_id: str

    document_id: int

    #
    # Hierarchy
    #

    article: str

    clause: str | None = None

    point: str | None = None

    #
    # Index
    #

    chunk_index: int

    sub_chunk_index: int = 0

    #
    # Character position
    #

    start_char: int

    end_char: int

    #
    # Metadata
    #

    title: str

    legal_type: str

    legal_sectors: str

    issuing_authority: str

    issuance_date: str

    #
    # Content
    #

    text: str