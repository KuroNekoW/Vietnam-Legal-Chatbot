from __future__ import annotations

from copy import deepcopy


class LengthSplitter:
    """
    Final splitter.

    Split long chunks into retrieval-sized chunks.

    Chunk
        ↓
    Length Split
        ↓
    Final Chunk
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
        chunk,
    ):

        text = chunk.text

        #
        # already short enough
        #

        if len(text) <= self.chunk_size:

            yield chunk
            return

        start = 0
        sub_index = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text),
            )

            new_chunk = deepcopy(chunk)

            new_chunk.text = text[start:end]

            new_chunk.start_char = (
                chunk.start_char + start
            )

            new_chunk.end_char = (
                chunk.start_char + end
            )

            new_chunk.sub_chunk_index = sub_index

            #
            # chunk id
            #

            new_chunk.chunk_id = (
                f"{chunk.chunk_id}_l{sub_index}"
            )

            yield new_chunk

            if end == len(text):
                break

            start = end - self.overlap
            sub_index += 1