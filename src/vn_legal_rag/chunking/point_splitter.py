from __future__ import annotations

from copy import deepcopy

from .regex import POINT_PATTERN


class PointSplitter:
    """
    Split one Clause into Points (Điểm).

    Input
    -----
    Clause chunk

    Output
    ------
    Nếu không có điểm:
        giữ nguyên.

    Nếu có:
        Khoản
            ├── a)
            ├── b)
            ├── c)
            └── ...
    """

    def split(
        self,
        clause_chunk,
    ):

        #
        # PREAMBLE
        #

        if clause_chunk.article_no is None:
            yield clause_chunk
            return

        text = clause_chunk.text

        matches = list(
            POINT_PATTERN.finditer(text)
        )

        #
        # Không có điểm
        #

        if not matches:
            yield clause_chunk
            return

        #
        # Có điểm
        #

        for i, match in enumerate(matches):

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(text)
            )

            point_text = text[start:end].strip()

            chunk = deepcopy(clause_chunk)

            point_no = match.group(1)

            chunk.point = f"{point_no})"
            chunk.point_no = point_no

            chunk.text = point_text

            chunk.start_char = start
            chunk.end_char = end

            #
            # chunk id
            #

            if chunk.clause_no is not None:

                chunk.chunk_id = (
                    f"{chunk.document_id}_"
                    f"{chunk.article_no}"
                    f"_c{chunk.clause_no}"
                    f"_p{point_no}"
                )

            else:

                chunk.chunk_id = (
                    f"{chunk.document_id}_"
                    f"{chunk.article_no}"
                    f"_p{point_no}"
                )

            yield chunk