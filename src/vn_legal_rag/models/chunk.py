from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Atomic retrieval unit.
    """

    # ==========================================================
    # Identity
    # ==========================================================

    chunk_id: str

    document_id: int

    # ==========================================================
    # Hierarchy
    # ==========================================================

    article: str
    clause: str | None = None
    point: str | None = None

    #
    # Parsed hierarchy
    #

    article_no: int | None = None
    clause_no: int | None = None
    point_no: str | None = None

    # ==========================================================
    # Chunk index
    # ==========================================================

    chunk_index: int
    sub_chunk_index: int = 0

    # ==========================================================
    # Character position
    # ==========================================================

    start_char: int
    end_char: int

    # ==========================================================
    # Metadata
    # ==========================================================

    title: str

    legal_type: str
    legal_sectors: str

    issuing_authority: str
    issuance_date: str

    url: str | None = None
    signers: str | None = None

    # ==========================================================
    # Content
    # ==========================================================

    text: str