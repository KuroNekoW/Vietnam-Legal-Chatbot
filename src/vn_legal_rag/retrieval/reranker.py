from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from vn_legal_rag.config import (
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MAX_LENGTH,
    RERANKER_MODEL,
)


class Reranker:
    """
    Vietnamese cross-encoder reranker.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str = RERANKER_DEVICE,
        batch_size: int = RERANKER_BATCH_SIZE,
        max_length: int = RERANKER_MAX_LENGTH,
    ):

        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):
            device = "cpu"

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        print(
            f"Loading reranker : {model_name}"
        )

        print(
            f"Reranker device   : {device}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
        )

        self.model.to(device)
        self.model.eval()

        print("Reranker loaded.")

    def rerank(
        self,
        query: str,
        chunks,
        top_k: int | None = None,
    ):

        if not chunks:
            return []

        query = query.strip()

        if not query:
            return []

        scored = []

        for start in range(
            0,
            len(chunks),
            self.batch_size,
        ):

            batch = chunks[
                start:start + self.batch_size
            ]

            pairs = [
                [
                    query,
                    chunk.text,
                ]
                for chunk in batch
            ]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                logits = self.model(
                    **inputs,
                    return_dict=True,
                ).logits

                scores = (
                    logits
                    .view(-1)
                    .float()
                    .cpu()
                    .tolist()
                )

            for chunk, score in zip(
                batch,
                scores,
            ):

                chunk.rerank_score = float(
                    score
                )

                scored.append(
                    chunk
                )

        scored.sort(
            key=lambda chunk: (
                chunk.rerank_score
            ),
            reverse=True,
        )

        if top_k is not None:

            scored = scored[:top_k]

        return scored