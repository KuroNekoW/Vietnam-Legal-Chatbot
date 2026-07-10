from typing import Generator

from vn_legal_rag.chunking.regex import CLAUSE_PATTERN
from vn_legal_rag.models import Chunk


class ClauseSplitter:
    """
    Split one Article into Clauses.

    Điều 5

    ↓

    Khoản 1

    Khoản 2

    Khoản 3
    """

    def split(
        self,
        chunk: Chunk,
    ) -> Generator[Chunk, None, None]:

        matches = list(
            CLAUSE_PATTERN.finditer(
                chunk.text
            )
        )

        #
        # No clause
        #

        if not matches:

            yield chunk

            return

        #
        # Split clauses
        #

        for i, match in enumerate(matches):

            start = match.start()

            if i == len(matches) - 1:

                end = len(
                    chunk.text
                )

            else:

                end = matches[
                    i + 1
                ].start()

            #
            # Keep article title with the first clause
            #

            if i == 0:

                text = chunk.text[
                    :end
                ].strip()

            else:

                text = chunk.text[
                    start:end
                ].strip()

            clause_title = match.group().strip()

            yield Chunk(

                chunk_id=f"{chunk.chunk_id}_c{i + 1}",

                document_id=chunk.document_id,

                article=chunk.article,

                clause=clause_title,

                point=None,

                chunk_index=chunk.chunk_index,

                sub_chunk_index=i + 1,

                start_char=chunk.start_char
                + (
                    0
                    if i == 0
                    else start
                ),

                end_char=chunk.start_char
                + end,

                title=chunk.title,

                legal_type=chunk.legal_type,

                legal_sectors=chunk.legal_sectors,

                issuing_authority=chunk.issuing_authority,

                issuance_date=chunk.issuance_date,

                text=text,
            )