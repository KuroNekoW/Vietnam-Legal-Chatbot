from __future__ import annotations

from pydantic import BaseModel, Field

from vn_legal_rag.generation import LegalAnswer


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=5000,
    )


class ChatResponse(BaseModel):
    query: str
    normalized_query: str | None

    retrieved_count: int
    reranked_count: int

    selected_document_count: int
    selected_chunk_count: int

    answer: LegalAnswer