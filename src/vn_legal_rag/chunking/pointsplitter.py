from typing import Generator

from vn_legal_rag.chunking.regex import POINT_PATTERN
from vn_legal_rag.models import Chunk


class PointSplitter:
    """
    Split one Clause into Points.

    Khoản

        a)

        b)

        c)

    ↓

    Point chunks
    """

    def split(
        self,
        chunk: Chunk,
    ) -> Generator[Chunk, None, None]:

        matches = list(
            POINT_PATTERN.finditer(
                chunk.text
            )
        )

        #
        # No point
        #

        if not matches:

            yield chunk

            return

        #
        # Split
        #

        for i, match in enumerate(matches):

            start = match.start()

            if i == len(matches) - 1:

                end = len(chunk.text)

            else:

                end = matches[
                    i + 1
                ].start()

            #
            # Keep heading with first point
            #

            if i == 0:

                text = chunk.text[
                    :end
                ].strip()

            else:

                text = chunk.text[
                    start:end
                ].strip()

            yield Chunk(

                chunk_id=f"{chunk.chunk_id}_p{i + 1}",

                document_id=chunk.document_id,

                article=chunk.article,

                clause=chunk.clause,

                point=match.group().strip(),

                chunk_index=chunk.chunk_index,

                sub_chunk_index=i + 1,

                start_char=chunk.start_char
                + (
                    0
                    if i == 0
                    else start
                ),

                end_char=chunk.start_char + end,

                title=chunk.title,

                legal_type=chunk.legal_type,

                legal_sectors=chunk.legal_sectors,

                issuing_authority=chunk.issuing_authority,

                issuance_date=chunk.issuance_date,

                text=text,
            )