from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnswerCitation(BaseModel):
    """
    Citation to a specific legal source used in the answer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    chunk_id: str | None = Field(
        default=None,
        description=(
            "ID của chunk chứa đoạn evidence được sử dụng."
        ),
    )

    document_id: int | None = None

    title: str = Field(
        min_length=1,
        description="Tên văn bản pháp luật được sử dụng.",
    )

    article: str | None = Field(
        default=None,
        description="Điều được trích dẫn.",
    )

    clause: str | None = Field(
        default=None,
        description="Khoản được trích dẫn.",
    )

    point: str | None = Field(
        default=None,
        description="Điểm được trích dẫn.",
    )

    issuance_date: str | None = Field(
        default=None,
        description="Ngày ban hành của văn bản.",
    )


class LegalAnswer(BaseModel):
    """
    Structured answer returned by the local LLM.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    answer: str = Field(
        min_length=1,
        description=(
            "Câu trả lời trực tiếp cho câu hỏi của người dùng, "
            "chỉ dựa trên context được cung cấp."
        ),
    )

    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description=(
            "Các nguồn pháp lý trong context được sử dụng "
            "để hỗ trợ câu trả lời."
        ),
    )

    insufficient_evidence: bool = Field(
        description=(
            "True nếu context không đủ căn cứ để trả lời "
            "một cách đáng tin cậy."
        ),
    )

    note: str | None = Field(
        default=None,
        description=(
            "Lưu ý về phạm vi áp dụng, ngoại lệ hoặc hạn chế "
            "của câu trả lời khi context có căn cứ."
        ),
    )


__all__ = [
    "AnswerCitation",
    "LegalAnswer",
]