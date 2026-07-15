from __future__ import annotations

from pathlib import Path

from vn_legal_rag.utils import append_chunk_index


class IndexBuilder:
    """
    Build FAISS index incrementally.

    Responsibilities
    ----------------
    - Encode one batch
    - Add vectors vào FAISS
    - Ghi metadata tương ứng

    Không chịu trách nhiệm:
    -----------------------
    - đọc file
    - progress bar
    - checkpoint
    - save/load faiss
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
        # Current FAISS id
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

        append_chunk_index(

            self.chunk_index_path,

            (

                {

                    #
                    # Vector id
                    #

                    "faiss_id": start_id + i,

                    #
                    # Chunk
                    #

                    "chunk_id": chunk.chunk_id,

                    "document_id": chunk.document_id,

                    #
                    # Hierarchy
                    #

                    "article_no": chunk.article_no,

                    "clause_no": chunk.clause_no,

                    "point_no": chunk.point_no,

                    "article": chunk.article,

                    "clause": chunk.clause,

                    "point": chunk.point,

                    #
                    # Position
                    #

                    "chunk_index": chunk.chunk_index,

                    "sub_chunk_index": chunk.sub_chunk_index,

                    "start_char": chunk.start_char,

                    "end_char": chunk.end_char,

                    #
                    # Metadata
                    #

                    "title": chunk.title,

                    "legal_type": chunk.legal_type,

                    "legal_sectors": chunk.legal_sectors,

                    "issuing_authority": chunk.issuing_authority,

                    "issuance_date": chunk.issuance_date,

                    "url": chunk.url,

                    "signers": chunk.signers,

                }

                for i, chunk in enumerate(chunks)

            ),

        )

        return len(chunks)

    @property
    def vectors(self) -> int:

        return self.faiss.ntotal