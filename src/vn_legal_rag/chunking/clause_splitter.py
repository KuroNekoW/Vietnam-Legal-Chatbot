from __future__ import annotations

from copy import deepcopy

from .regex import CLAUSE_PATTERN


class ClauseSplitter:
    """
    Split one Article into Clauses (Khoản).

    Responsibilities
    -----------------
    - Detect clauses
    - Set clause metadata
    - Maintain character positions relative to the Article

    This splitter does NOT generate the final chunk_id.
    """

    def split(
        self,
        article_chunk,
    ):

        # ==================================================
        # Preamble
        # ==================================================

        if article_chunk.article_no is None:

            yield article_chunk

            return

        text = article_chunk.text

        matches = list(
            CLAUSE_PATTERN.finditer(text)
        )

        # ==================================================
        # No clauses
        # ==================================================

        if not matches:

            yield article_chunk

            return

        # ==================================================
        # Clauses
        # ==================================================

        for i, match in enumerate(matches):

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            clause_text = text[
                start:end
            ].strip()

            chunk = deepcopy(
                article_chunk
            )

            clause_no = int(
                match.group(1)
            )

            # ------------------------------------------------
            # Hierarchy
            # ------------------------------------------------

            chunk.clause = (
                f"Khoản {clause_no}"
            )

            chunk.clause_no = clause_no

            # ------------------------------------------------
            # Content
            # ------------------------------------------------

            chunk.text = clause_text

            # ------------------------------------------------
            # Position
            #
            # Keep coordinates relative to the Article.
            # ------------------------------------------------

            leading_offset = (
                len(text[start:end])
                - len(text[start:end].lstrip())
            )

            chunk.start_char = (
                start + leading_offset
            )

            chunk.end_char = (
                chunk.start_char
                + len(clause_text)
            )

            # ------------------------------------------------
            # IMPORTANT
            #
            # Do NOT modify chunk.chunk_id here.
            # ------------------------------------------------

            yield chunk