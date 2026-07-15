from pydantic import BaseModel


class Chunk(BaseModel):
    """
    Atomic retrieval unit.

    One chunk corresponds to one searchable unit in the
    FAISS vector database.
    """

    # ==========================================================
    # Identity
    # ==========================================================

    chunk_id: str
    document_id: int

    # ==========================================================
    # Legal hierarchy
    # ==========================================================

    article: str
    article_no: int | None = None

    clause: str | None = None
    clause_no: int | None = None

    point: str | None = None
    point_no: str | None = None

    # ==========================================================
    # Chunk position
    # ==========================================================

    chunk_index: int
    sub_chunk_index: int = 0

    start_char: int
    end_char: int

    # ==========================================================
    # Document metadata
    # ==========================================================

    title: str

    legal_type: str
    legal_sectors: str

    issuing_authority: str
    issuance_date: str

    url: str | None = None
    signers: str | None = None

    # ==========================================================
    # Retrieval content
    # ==========================================================

    text: str