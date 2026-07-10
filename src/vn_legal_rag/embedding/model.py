from __future__ import annotations

import numpy as np
import torch

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper của SentenceTransformer.

    Responsibilities
    ----------------
    - Load embedding model
    - Encode single text
    - Encode batch
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        print(f"Embedding device: {device}")

        #
        # TensorFloat32 (RTX30/40)
        #
        if device == "cuda":

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        #
        # Load model
        #
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )

    @property
    def dimension(self) -> int:
        """
        Embedding dimension.
        """

        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        text: str,
    ) -> np.ndarray:

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 512,
    ) -> np.ndarray:

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )