from typing import Generator

from vn_legal_rag.models import Chunk


class LengthSplitter:
    """
    Final fallback splitter.

    Chỉ dùng khi một semantic chunk vẫn quá dài.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        chunk: Chunk,
    ) -> Generator[Chunk, None, None]:

        text = chunk.text

        #
        # Already small enough
        #

        if len(text) <= self.chunk_size:

            yield chunk
            return

        start = 0
        sub_index = 0

        while start < len(text):

            #
            # Candidate end
            #

            end = min(
                start + self.chunk_size,
                len(text),
            )

            #
            # Find better split point
            #

            if end < len(text):

                better_end = self._find_split_position(
                    text,
                    start,
                    end,
                )

                if better_end > start:

                    end = better_end

            #
            # Output chunk
            #

            yield Chunk(

                chunk_id=f"{chunk.chunk_id}_l{sub_index}",

                document_id=chunk.document_id,

                article=chunk.article,

                clause=chunk.clause,

                point=chunk.point,

                chunk_index=chunk.chunk_index,

                sub_chunk_index=sub_index,

                start_char=chunk.start_char + start,

                end_char=chunk.start_char + end,

                title=chunk.title,

                legal_type=chunk.legal_type,

                legal_sectors=chunk.legal_sectors,

                issuing_authority=chunk.issuing_authority,

                issuance_date=chunk.issuance_date,

                text=text[start:end].strip(),
            )

            #
            # Finished
            #

            if end >= len(text):

                break

            #
            # Overlap
            #

            start = max(
                end - self.overlap,
                0,
            )

            #
            # Don't begin in the middle of a word
            #

            start = self._move_to_word_boundary(
                text,
                start,
            )

            sub_index += 1

    def _find_split_position(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """
        Prefer semantic separators.
        """

        window = text[start:end]

        separators = [

            "\n\n",

            "\n",

            ";",

            ".",

            ",",

            " ",

        ]

        for sep in separators:

            pos = window.rfind(sep)

            #
            # Ignore separators too close
            # to beginning of chunk
            #

            if pos > self.chunk_size // 2:

                return start + pos + len(sep)

        return end

    def _move_to_word_boundary(
        self,
        text: str,
        start: int,
    ) -> int:
        """
        Move overlap start so it begins
        at a complete word.
        """

        if start <= 0:

            return 0

        #
        # Skip remaining letters
        #

        while (

            start < len(text)

            and not text[start].isspace()

        ):

            start += 1

        #
        # Skip spaces/newlines
        #

        while (

            start < len(text)

            and text[start].isspace()

        ):

            start += 1

        return start