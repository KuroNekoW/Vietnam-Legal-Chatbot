from __future__ import annotations


class IndexBuilder:
    """
    Encode chunks rồi đưa vào Vector Store.

    Embedding text chứa contextual hierarchy:

        Article
        Clause
        Point
        Chunk text

    nhưng chunk.text gốc không bị thay đổi.
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

    # =========================================================
    # Build embedding text
    # =========================================================

    @staticmethod
    def build_embedding_text(
        chunk,
    ) -> str:
        """
        Tạo text dùng riêng cho embedding.

        Không thay đổi chunk.text.

        Ví dụ:

            Điều 8. Các hành vi bị nghiêm cấm
            Khoản 17
            Điểm a)
            Không được ...

        """

        parts: list[str] = []

        # -----------------------------------------------------
        # Article
        # -----------------------------------------------------

        if chunk.article:

            parts.append(
                chunk.article.strip()
            )

        # -----------------------------------------------------
        # Clause
        # -----------------------------------------------------

        if chunk.clause:

            parts.append(
                chunk.clause.strip()
            )

        # -----------------------------------------------------
        # Point
        # -----------------------------------------------------

        if chunk.point:

            parts.append(
                chunk.point.strip()
            )

        # -----------------------------------------------------
        # Actual chunk content
        # -----------------------------------------------------

        if chunk.text:

            parts.append(
                chunk.text.strip()
            )

        return "\n".join(
            part
            for part in parts
            if part
        )

    # =========================================================
    # Process batch
    # =========================================================

    def process_batch(
        self,
        chunks,
    ) -> int:

        if not chunks:
            return 0

        # -----------------------------------------------------
        # Contextual embedding text
        # -----------------------------------------------------

        texts = [
            self.build_embedding_text(
                chunk
            )
            for chunk in chunks
        ]

        # -----------------------------------------------------
        # Encode
        # -----------------------------------------------------

        embeddings = (
            self.embedding_model.encode_batch(
                texts,
                batch_size=self.batch_size,
            )
        )

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        self.store.add(
            embeddings=embeddings,
            chunks=chunks,
        )

        return len(chunks)

    # =========================================================
    # Statistics
    # =========================================================

    @property
    def vectors(self):

        return self.store.ntotal