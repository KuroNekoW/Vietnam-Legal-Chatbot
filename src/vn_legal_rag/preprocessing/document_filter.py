from __future__ import annotations


class DocumentFilter:

    #
    # Chỉ giữ các loại văn bản này
    #

    ALLOWED_TYPES = {

        "hiến pháp",
        "luật",
        "văn bản hợp nhất",
        "pháp lệnh",
        "nghị định",
    }
 
    @classmethod
    def check(
        cls,
        document,
    ) -> tuple[bool, str]:

        legal_type = (
            document.legal_type or ""
        ).strip().lower()

        content = (
            document.content or ""
        ).strip()

        #
        # legal type
        #

        if legal_type not in cls.ALLOWED_TYPES:

            return (
                False,
                f"type:{legal_type or 'unknown'}",
            )

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
    def should_keep(
        cls,
        document,
    ):

        return cls.check(document)[0]