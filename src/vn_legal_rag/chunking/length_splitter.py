from __future__ import annotations

from copy import deepcopy
import hashlib


class LengthSplitter:
    """
    Final splitter.

    Split long chunks into retrieval-sized chunks.

    Responsibilities
    -----------------
    - Split chunks by length
    - Maintain final character positions
    - Generate the FINAL globally unique chunk_id

    The final chunk_id is generated only here.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
    ):

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be > 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap must be >= 0"
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    # ======================================================
    # Public
    # ======================================================

    def split(
        self,
        chunk,
    ):

        text = chunk.text

        # ==================================================
        # Already short enough
        # ==================================================

        if len(text) <= self.chunk_size:

            final_chunk = deepcopy(
                chunk
            )

            final_chunk.sub_chunk_index = 0

            final_chunk.chunk_id = (
                self._generate_chunk_id(
                    final_chunk
                )
            )

            yield final_chunk

            return

        # ==================================================
        # Length split
        # ==================================================

        start = 0
        sub_index = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk_text = text[
                start:end
            ]

            new_chunk = deepcopy(
                chunk
            )

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            new_chunk.text = chunk_text

            # ------------------------------------------------
            # Position
            #
            # chunk.start_char is relative to Article.
            # ------------------------------------------------

            new_chunk.start_char = (
                chunk.start_char
                + start
            )

            new_chunk.end_char = (
                chunk.start_char
                + end
            )

            # ------------------------------------------------
            # Sub chunk index
            # ------------------------------------------------

            new_chunk.sub_chunk_index = (
                sub_index
            )

            # ------------------------------------------------
            # FINAL ID
            # ------------------------------------------------

            new_chunk.chunk_id = (
                self._generate_chunk_id(
                    new_chunk
                )
            )

            yield new_chunk

            # ------------------------------------------------
            # Finished
            # ------------------------------------------------

            if end == len(text):

                break

            # ------------------------------------------------
            # Overlap
            # ------------------------------------------------

            start = (
                end - self.overlap
            )

            sub_index += 1

    # ======================================================
    # Generate final chunk ID
    # ======================================================

    @staticmethod
    def _generate_chunk_id(
        chunk,
    ) -> str:
        """
        Generate a deterministic final chunk ID.

        The ID is based on:

        - document
        - article
        - clause
        - point
        - chunk position
        - sub chunk position
        - text content

        This prevents different chunks from accidentally
        sharing the same ID.
        """

        identity = "|".join(
            [
                str(chunk.document_id),

                str(
                    chunk.article_no
                    if chunk.article_no is not None
                    else ""
                ),

                str(
                    chunk.clause_no
                    if chunk.clause_no is not None
                    else ""
                ),

                str(
                    chunk.point_no
                    if chunk.point_no is not None
                    else ""
                ),

                str(chunk.chunk_index),

                str(chunk.sub_chunk_index),

                str(chunk.start_char),

                str(chunk.end_char),

                chunk.text,
            ]
        )

        digest = hashlib.blake2b(
            identity.encode(
                "utf-8"
            ),
            digest_size=16,
        ).hexdigest()

        return digest