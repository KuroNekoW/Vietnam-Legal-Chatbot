from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NormalizedQuery(BaseModel):
    """
    Structured result for retrieval-oriented query normalization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    normalized_query: str = Field(
        min_length=1,
        description=(
            "Câu truy vấn pháp lý tối ưu cho semantic retrieval. "
            "Phải chứa các thuật ngữ pháp lý cốt lõi, đối tượng, "
            "hành vi/vấn đề pháp lý, và các ràng buộc quan trọng "
            "được nêu trong câu hỏi gốc."
        ),
    )

    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Các từ khóa pháp lý cốt lõi phải xuất hiện trực tiếp "
            "trong normalized_query và có giá trị cao cho retrieval."
        ),
    )

    legal_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Các thuật ngữ pháp lý chuẩn hoặc thuật ngữ chính thức "
            "liên quan trực tiếp đến câu hỏi."
        ),
    )

    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Các điều kiện, tình huống, chủ thể, loại giấy tờ, "
            "ngoại lệ hoặc chi tiết quan trọng phải được bảo toàn."
        ),
    )

    temporal_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Mọi thông tin thời gian được nêu trong câu hỏi, "
            "ví dụ năm, ngày, khoảng thời gian, thời điểm phát sinh."
        ),
    )