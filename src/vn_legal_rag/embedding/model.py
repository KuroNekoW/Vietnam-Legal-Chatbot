from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """
    Wrapper của SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ):

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )

    def encode(
        self,
        text: str,
    ) -> np.ndarray:

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )