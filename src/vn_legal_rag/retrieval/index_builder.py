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

    Does NOT
    --------
    - iterate dataset
    - display progress
    - save/load index
    - checkpoint
    """

    def __init__(
        self,
        embedding_model,
        faiss_index,
        chunk_index_path: str | Path,
    ):

        self.embedding_model = embedding_model
        self.faiss = faiss_index
        self.chunk_index_path = Path(chunk_index_path)

    def process_batch(
        self,
        chunks,
        batch_size: int = 512,
    ) -> int:
        """
        Process ONE batch.

        Parameters
        ----------
        chunks
            list[Chunk]

        Returns
        -------
        int
            Number of vectors added.
        """

        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode_batch(
            texts,
            batch_size=batch_size,
        )

        self.faiss.add(
            embeddings
        )

        append_chunk_index(
            self.chunk_index_path,
            (
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "article": chunk.article,
                    "legal_type": chunk.legal_type,
                    "legal_sectors": chunk.legal_sectors,
                    "issuing_authority": chunk.issuing_authority,
                    "issuance_date": chunk.issuance_date,
                }
                for chunk in chunks
            ),
        )

        return len(chunks)

    @property
    def vectors(
        self,
    ) -> int:
        """
        Current number of vectors.
        """

        return self.faiss.ntotal