from typing import Generator

from vn_legal_rag.models import Chunk


class LengthSplitter:

    def __init__(

        self,

        chunk_size=1000,

        overlap=200,

    ):

        self.chunk_size = chunk_size

        self.overlap = overlap

    def split(

        self,

        chunk: Chunk,

    ) -> Generator[Chunk, None, None]:

        text = chunk.text

        #
        # small chunk
        #

        if len(text) <= self.chunk_size:

            yield chunk

            return

        start = 0

        index = 0

        while start < len(text):

            end = min(

                start + self.chunk_size,

                len(text),

            )

            yield Chunk(

                chunk_id=f"{chunk.chunk_id}_{index}",

                document_id=chunk.document_id,

                article=chunk.article,

                chunk_index=index,

                start_char=chunk.start_char + start,

                end_char=chunk.start_char + end,

                title=chunk.title,

                legal_type=chunk.legal_type,

                legal_sectors=chunk.legal_sectors,

                issuing_authority=chunk.issuing_authority,

                issuance_date=chunk.issuance_date,

                text=text[start:end],
            )

            if end == len(text):

                break

            start = end - self.overlap

            index += 1