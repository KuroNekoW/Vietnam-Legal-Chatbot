from __future__ import annotations

from collections import Counter


class DocumentFilter:

    BLACKLIST_TITLE = (
        "bộ câu hỏi",
        "ngân hàng câu hỏi",
        "đề thi",
        "đáp án",
        "câu hỏi sát hạch",
        "bộ đề",
    )

    BLACKLIST_TYPE = (
        # thêm sau nếu cần
    )

    @classmethod
    def check(
        cls,
        document,
    ) -> tuple[bool, str]:

        title = (document.title or "").lower()
        legal_type = (document.legal_type or "").lower()
        content = (document.content or "").strip()

        #
        # blacklist title
        #

        for keyword in cls.BLACKLIST_TITLE:

            if keyword in title:

                return False, f"title:{keyword}"

        #
        # blacklist legal type
        #

        for keyword in cls.BLACKLIST_TYPE:

            if keyword in legal_type:

                return False, f"type:{keyword}"

        #
        # empty
        #

        if not content:

            return False, "empty"

        #
        # too short
        #

        if len(content) < 100:

            return False, "too_short"

        return True, "ok"

    @classmethod
    def should_keep(cls, document):

        return cls.check(document)[0]