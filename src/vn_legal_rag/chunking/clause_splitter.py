from __future__ import annotations

from copy import deepcopy

from .regex import CLAUSE_PATTERN


class ClauseSplitter:
    """
    Split one Article into Clauses (Khoản).

    Input
    -----
    PREAMBLE hoặc Điều

    Output
    ------
    Nếu không có khoản:
        giữ nguyên.

    Nếu có khoản:
        Điều
            ├── Khoản 1
            ├── Khoản 2
            └── ...
    """

    def split(
        self,
        article_chunk,
    ):

        #
        # PREAMBLE không tách
        #

        if article_chunk.article_no is None:
            yield article_chunk
            return

        text = article_chunk.text

        matches = list(
            CLAUSE_PATTERN.finditer(text)
        )

        #
        # Không có khoản
        #

        if not matches:
            yield article_chunk
            return

        #
        # Có khoản
        #

        for i, match in enumerate(matches):

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            clause_text = text[start:end].strip()

            chunk = deepcopy(article_chunk)

            clause_no = int(match.group(1))

            chunk.clause = f"Khoản {clause_no}"
            chunk.clause_no = clause_no

            chunk.text = clause_text

            chunk.start_char = start
            chunk.end_char = end

            chunk.chunk_id = (
                f"{article_chunk.chunk_id}"
                f"_c{clause_no}"
            )

            yield chunk