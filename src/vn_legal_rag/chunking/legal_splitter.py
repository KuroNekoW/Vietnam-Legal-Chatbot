from __future__ import annotations

from vn_legal_rag.models import Chunk
from .regex import (
    ARTICLE_PATTERN,
    PREAMBLE_TITLE,
)


class LegalSplitter:
    """
    Split a legal document into Articles.

    Output:
        PREAMBLE
        Điều 1
        Điều 2
        ...
    """

    def split(
        self,
        document,
    ):

        content = document.content.strip()

        matches = list(
            ARTICLE_PATTERN.finditer(content)
        )

        #
        # No article
        #

        if not matches:

            yield self._build_chunk(
                document=document,
                article=PREAMBLE_TITLE,
                article_no=None,
                text=content,
                chunk_index=0,
            )

            return

        #
        # Preamble
        #

        preamble = content[
            :matches[0].start()
        ].strip()

        chunk_index = 0

        if preamble:

            yield self._build_chunk(
                document=document,
                article=PREAMBLE_TITLE,
                article_no=None,
                text=preamble,
                chunk_index=chunk_index,
            )

            chunk_index += 1

        #
        # Articles
        #

        for i, match in enumerate(matches):

            start = match.start()

            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(content)
            )

            article_text = content[start:end].strip()

            article_title = (
                f"Điều {match.group(1)}. {match.group(2).strip()}"
            )

            yield self._build_chunk(
                document=document,
                article=article_title,
                article_no=int(match.group(1)),
                text=article_text,
                chunk_index=chunk_index,
            )

            chunk_index += 1

    def _build_chunk(
        self,
        document,
        article,
        article_no,
        text,
        chunk_index,
    ) -> Chunk:

        return Chunk(

            #
            # Identity
            #

            chunk_id=f"{document.id}_{chunk_index}",

            document_id=document.id,

            #
            # Hierarchy
            #

            article=article,
            article_no=article_no,

            clause=None,
            clause_no=None,

            point=None,
            point_no=None,

            #
            # Position
            #

            chunk_index=chunk_index,
            sub_chunk_index=0,

            start_char=0,
            end_char=len(text),

            #
            # Metadata
            #

            title=document.title,

            legal_type=document.legal_type,
            legal_sectors=document.legal_sectors,

            issuing_authority=document.issuing_authority,
            issuance_date=document.issuance_date,

            url=document.url,
            signers=document.signers,

            #
            # Content
            #

            text=text,
        )