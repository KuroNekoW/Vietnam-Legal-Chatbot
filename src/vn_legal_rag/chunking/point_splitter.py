from __future__ import annotations

from copy import deepcopy

from .regex import POINT_PATTERN


class PointSplitter:
    """
    Split one Clause into Points (Điểm).

    Responsibilities
    -----------------
    - Detect points
    - Set point metadata
    - Maintain character positions relative to the Article

    This splitter does NOT generate the final chunk_id.
    """

    def split(
        self,
        clause_chunk,
    ):

        # ==================================================
        # Preamble
        # ==================================================

        if clause_chunk.article_no is None:

            yield clause_chunk

            return

        text = clause_chunk.text

        matches = list(
            POINT_PATTERN.finditer(text)
        )

        # ==================================================
        # No points
        # ==================================================

        if not matches:

            yield clause_chunk

            return

        # ==================================================
        # Points
        # ==================================================

        for i, match in enumerate(matches):

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            point_text = text[
                start:end
            ].strip()

            chunk = deepcopy(
                clause_chunk
            )

            point_no = match.group(1)

            # ------------------------------------------------
            # Hierarchy
            # ------------------------------------------------

            chunk.point = (
                f"{point_no})"
            )

            chunk.point_no = point_no

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            chunk.text = point_text

            # ------------------------------------------------
            # Position
            #
            # clause_chunk.start_char is already relative
            # to the Article.
            #
            # Therefore convert point position to
            # Article-relative coordinates.
            # ------------------------------------------------

            leading_offset = (
                len(text[start:end])
                - len(text[start:end].lstrip())
            )

            relative_start = (
                start + leading_offset
            )

            chunk.start_char = (
                clause_chunk.start_char
                + relative_start
            )

            chunk.end_char = (
                chunk.start_char
                + len(point_text)
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # Do NOT modify chunk.chunk_id here.
            # ------------------------------------------------

            yield chunk