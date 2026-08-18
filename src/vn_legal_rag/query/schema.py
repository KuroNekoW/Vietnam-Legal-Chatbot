from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NormalizedQuery(BaseModel):
    """
    Structured representation of a Vietnamese legal query.

    The model should preserve the meaning and all legally relevant
    information from the original user query.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    normalized_query: str = Field(
        min_length=1,
        description=(
            "Bản viết lại đầy đủ của câu hỏi người dùng bằng "
            "văn phong pháp lý tự nhiên. Không được rút gọn "
            "thành danh sách từ khóa. Phải giữ nguyên ý nghĩa, "
            "đối tượng, hành vi, điều kiện, mốc thời gian, "
            "số lượng và câu hỏi mà người dùng đang hỏi."
        ),
    )

    question_intent: str = Field(
        min_length=1,
        description=(
            "Ý định pháp lý chính của câu hỏi, ví dụ: "
            "thời hạn báo trước, điều kiện cấp giấy chứng nhận, "
            "trình tự thủ tục, quyền và nghĩa vụ, mức xử phạt."
        ),
    )

    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "3 đến 8 từ khóa/cụm từ pháp lý có giá trị cao "
            "cho retrieval. Có thể là thuật ngữ pháp lý, "
            "hành vi, đối tượng, loại giấy tờ, thời hạn, "
            "mốc thời gian hoặc điều kiện đặc thù."
        ),
    )

    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Các thông tin ràng buộc hoặc tình tiết quan trọng "
            "của câu hỏi phải được bảo toàn."
        ),
    )

    temporal_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Các thông tin về thời gian, thời hạn, ngày, tháng, "
            "năm hoặc khoảng thời gian xuất hiện trong câu hỏi."
        ),
    )