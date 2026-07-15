from __future__ import annotations

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from vn_legal_rag.config import EMBEDDING_MODEL


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.

    Responsibilities
    ----------------
    - Load embedding model
    - Encode one text
    - Encode batch
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        print(f"Embedding device : {device}")
        print(f"Embedding model  : {model_name}")

        #
        # TensorFloat32
        #

        if device == "cuda":

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            if hasattr(
                torch,
                "set_float32_matmul_precision",
            ):

                torch.set_float32_matmul_precision(
                    "high"
                )

        #
        # Load model
        #

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )

    @property
    def dimension(
        self,
    ) -> int:

        return self.model.get_sentence_embedding_dimension()

    #
    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------
    #

    def _prepare_text(
        self,
        text: str,
        text_type: str,
    ) -> str:

        #
        # E5 requires prefixes
        #

        if "e5" in self.model_name.lower():

            return f"{text_type}: {text}"

        #
        # Other embedding models
        #

        return text

    #
    # ---------------------------------------------------------
    # Encode single
    # ---------------------------------------------------------
    #

    def encode(
        self,
        text: str,
        text_type: str = "query",
    ) -> np.ndarray:

        text = self._prepare_text(
            text,
            text_type,
        )

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    #
    # ---------------------------------------------------------
    # Encode batch
    # ---------------------------------------------------------
    #

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 512,
        text_type: str = "passage",
    ) -> np.ndarray:

        texts = [

            self._prepare_text(
                text,
                text_type,
            )

            for text in texts

        ]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )