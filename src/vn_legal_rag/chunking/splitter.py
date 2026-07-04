import re
from typing import Generator

from vn_legal_rag.chunking.regex import ARTICLE_PATTERN
from vn_legal_rag.models import Chunk
from vn_legal_rag.models import LegalDocument


class LegalSplitter:
    """
    Split Vietnamese legal documents into semantic chunks.

    Strategy

    Document

    ↓

    PREAMBLE

    ↓

    Điều 1

    ↓

    Điều 2

    ↓

    ...
    """

    def split(
        self,
        document: LegalDocument,
    ) -> Generator[Chunk, None, None]:

        text = document.content

        matches = list(
            ARTICLE_PATTERN.finditer(text)
        )

        chunk_index = 0

        #
        # No article
        #

        if not matches:

            yield Chunk(

                chunk_id=f"{document.id}_0",

                document_id=document.id,

                article="FULL_DOCUMENT",

                chunk_index=0,

                start_char=0,

                end_char=len(text),

                title=document.title,

                legal_type=document.legal_type,

                legal_sectors=document.legal_sectors,

                issuing_authority=document.issuing_authority,

                issuance_date=document.issuance_date,

                text=text.strip(),
            )

            return

        #
        # PREAMBLE
        #

        first_start = matches[0].start()

        preamble = text[:first_start].strip()

        if preamble:

            yield Chunk(

                chunk_id=f"{document.id}_0",

                document_id=document.id,

                article="PREAMBLE",

                chunk_index=0,

                start_char=0,

                end_char=first_start,

                title=document.title,

                legal_type=document.legal_type,

                legal_sectors=document.legal_sectors,

                issuing_authority=document.issuing_authority,

                issuance_date=document.issuance_date,

                text=preamble,
            )

            chunk_index += 1

        #
        # Articles
        #

        for i, match in enumerate(matches):

            start = match.start()

            if i == len(matches) - 1:

                end = len(text)

            else:

                end = matches[i + 1].start()

            article_text = text[start:end].strip()

            article_title = match.group().strip()

            yield Chunk(

                chunk_id=f"{document.id}_{chunk_index}",

                document_id=document.id,

                article=article_title,

                chunk_index=chunk_index,

                start_char=start,

                end_char=end,

                title=document.title,

                legal_type=document.legal_type,

                legal_sectors=document.legal_sectors,

                issuing_authority=document.issuing_authority,

                issuance_date=document.issuance_date,

                text=article_text,
            )

            chunk_index += 1