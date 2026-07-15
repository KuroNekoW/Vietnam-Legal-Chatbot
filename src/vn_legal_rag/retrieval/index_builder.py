from __future__ import annotations

from pathlib import Path

from vn_legal_rag.utils import append_chunk_index


class IndexBuilder:
    """
    Build FAISS index incrementally.

    Responsibilities
    ----------------
    - Encode one batch
    - Add vectors into FAISS
    - Save metadata
    """

    def __init__(
        self,
        embedding_model,
        faiss_index,
        chunk_index_path: str | Path,
        batch_size: int = 512,
    ):

        self.embedding_model = embedding_model
        self.faiss = faiss_index
        self.chunk_index_path = Path(chunk_index_path)
        self.batch_size = batch_size

    def process_batch(
        self,
        chunks,
    ) -> int:

        if not chunks:
            return 0

        #
        # Encode
        #

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode_batch(
            texts,
            batch_size=self.batch_size,
        )

        #
        # Current FAISS position
        #

        start_id = self.faiss.ntotal

        #
        # Add vectors
        #

        self.faiss.add(
            embeddings
        )

        #
        # Save metadata
        #

        rows = []

        for offset, chunk in enumerate(chunks):

            rows.append(

                {

                    #
                    # FAISS
                    #

                    "faiss_id": start_id + offset,

                    #
                    # Chunk
                    #

                    "chunk_id": chunk.chunk_id,

                    #
                    # Document
                    #

                    "document_id": chunk.document_id,

                    #
                    # Hierarchy
                    #

                    "article_no": getattr(
                        chunk,
                        "article_no",
                        None,
                    ),

                    "clause_no": getattr(
                        chunk,
                        "clause_no",
                        None,
                    ),

                    "point": getattr(
                        chunk,
                        "point",
                        None,
                    ),

                    #
                    # Human-readable
                    #

                    "article": chunk.article,

                    "title": chunk.title,

                    #
                    # Metadata
                    #

                    "legal_type": chunk.legal_type,

                    "legal_sectors": chunk.legal_sectors,

                    "issuing_authority": chunk.issuing_authority,

                    "issuance_date": chunk.issuance_date,

                    "url": getattr(
                        chunk,
                        "url",
                        None,
                    ),

                    "signers": getattr(
                        chunk,
                        "signers",
                        None,
                    ),

                }

            )

        append_chunk_index(
            self.chunk_index_path,
            rows,
        )

        return len(chunks)

    @property
    def vectors(self):

        return self.faiss.ntotal