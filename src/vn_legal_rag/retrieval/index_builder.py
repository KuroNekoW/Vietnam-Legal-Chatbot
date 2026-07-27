from __future__ import annotations


class IndexBuilder:
    """
    Encode chunks rồi đưa vào Vector Store.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        batch_size: int = 512,
    ):

        self.embedding_model = embedding_model
        self.store = vector_store
        self.batch_size = batch_size

    def process_batch(
        self,
        chunks,
    ) -> int:

        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode_batch(
            texts,
            batch_size=self.batch_size,
        )

        self.store.add(
            embeddings=embeddings,
            chunks=chunks,
        )

        return len(chunks)

    @property
    def vectors(self):

        return self.store.ntotal