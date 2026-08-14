from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NormalizedQuery(BaseModel):
    """
    Structured result produced by the local LLM
    for legal query normalization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    normalized_query: str = Field(
        min_length=1,
        description=(
            "Câu hỏi được chuẩn hóa bằng thuật ngữ pháp lý "
            "rõ ràng nhưng giữ nguyên ý nghĩa và các thông tin "
            "quan trọng của câu hỏi gốc."
        ),
    )

    legal_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Các thuật ngữ pháp lý quan trọng phục vụ "
            "semantic retrieval."
        ),
    )

    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Các điều kiện, mốc thời gian, chủ thể, giấy tờ, "
            "ngoại lệ hoặc tình tiết quan trọng phải được giữ lại."
        ),
    )