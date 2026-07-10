from __future__ import annotations

from tqdm import tqdm

from vn_legal_rag.utils import append_chunk_index


class IndexBuilder:
    """
    Build FAISS index from streamed chunks.

    Pipeline

    Chunks
        ↓
    Batch
        ↓
    Embedding
        ↓
    FAISS.add()
        ↓
    chunk_index.jsonl
    """

    def __init__(
        self,
        embedding_model,
        faiss_index,
        chunk_index_path,
        batch_size: int = 512,
    ):

        self.embedding_model = embedding_model
        self.faiss = faiss_index
        self.chunk_index_path = chunk_index_path
        self.batch_size = batch_size

    def build(
        self,
        chunks,
        total_chunks: int | None = None,
    ):

        texts = []
        metadata = []

        progress = tqdm(
            total=total_chunks,
            desc="Indexing",
            unit="chunk",
            colour="green",
            dynamic_ncols=True,
        )

        for chunk in chunks:

            texts.append(
                chunk.text
            )

            metadata.append(
                chunk
            )

            if len(texts) >= self.batch_size:

                self._flush(
                    texts,
                    metadata,
                )

                progress.update(
                    len(texts)
                )

                texts.clear()
                metadata.clear()

        #
        # remaining
        #

        if texts:

            self._flush(
                texts,
                metadata,
            )

            progress.update(
                len(texts)
            )

        progress.close()

    def _flush(
        self,
        texts,
        metadata,
    ):

        embeddings = self.embedding_model.encode_batch(
            texts,
            batch_size=self.batch_size,
        )

        self.faiss.add(
            embeddings
        )

        rows = []

        for chunk in metadata:

            rows.append(

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

            )

        append_chunk_index(
            self.chunk_index_path,
            rows,
        )