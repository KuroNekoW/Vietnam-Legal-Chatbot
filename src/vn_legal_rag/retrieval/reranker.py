from __future__ import annotations

import re
import unicodedata

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

    Pipeline
    --------
    Qdrant candidates
        ↓
    Cross-encoder scoring
        ↓
    Sort by rerank score
        ↓
    Deduplicate equivalent chunks
        ↓
    Top-K
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

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
            )
        )

        self.model.to(device)
        self.model.eval()

        print("Reranker loaded.")

    # =========================================================
    # Rerank
    # =========================================================

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

        # -----------------------------------------------------
        # Score candidates in batches
        # -----------------------------------------------------

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
                key: value.to(
                    self.device
                )
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

        # -----------------------------------------------------
        # Sort by reranker score
        # -----------------------------------------------------

        scored.sort(
            key=lambda chunk: (
                chunk.rerank_score
                if chunk.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        # -----------------------------------------------------
        # Deduplicate
        # -----------------------------------------------------

        scored, duplicate_groups = (
            self._deduplicate(
                scored
            )
        )

        # -----------------------------------------------------
        # Top-K AFTER dedup
        # -----------------------------------------------------

        if top_k is not None:

            scored = scored[
                :top_k
            ]

        # Store information so the test/debug code can inspect it.
        self.last_duplicate_groups = (
            duplicate_groups
        )

        return scored

    # =========================================================
    # Deduplication
    # =========================================================

    def _deduplicate(
        self,
        chunks,
    ):
        """
        Loại các chunk có cùng nội dung pháp lý.

        Signature không chứa document_id hoặc title,
        vì nhiều phiên bản văn bản có thể chứa cùng một
        quy định pháp lý.

        Ví dụ:

            VBHN 2025
            VBHN 2026
            Bộ luật 2019

        cùng Điều 36 Khoản 3 và cùng text
        → chỉ giữ một representative.
        """

        seen = {}

        duplicate_groups = []

        for chunk in chunks:

            signature = (
                chunk.article_no,
                chunk.clause_no,
                chunk.point_no,
                self._normalize_text(
                    chunk.text
                ),
            )

            existing = seen.get(
                signature
            )

            # -------------------------------------------------
            # First occurrence
            # -------------------------------------------------

            if existing is None:

                seen[signature] = chunk

                continue

            # -------------------------------------------------
            # Duplicate
            # -------------------------------------------------

            winner, duplicate = (
                self._choose_representative(
                    existing,
                    chunk,
                )
            )

            seen[signature] = winner

            duplicate_groups.append(
                {
                    "kept": winner,
                    "removed": duplicate,
                }
            )

        #
        # Important:
        # giữ đúng thứ tự score sau dedup.
        #

        deduplicated = sorted(
            seen.values(),
            key=lambda chunk: (
                chunk.rerank_score
                if chunk.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        return (
            deduplicated,
            duplicate_groups,
        )

    # =========================================================
    # Representative selection
    # =========================================================

    @staticmethod
    def _choose_representative(
        first,
        second,
    ):
        """
        Chọn chunk đại diện.

        Priority:
        1. rerank score cao hơn
        2. issuance_date mới hơn nếu score bằng nhau
        3. giữ first nếu vẫn bằng nhau
        """

        first_score = (
            first.rerank_score
            if first.rerank_score is not None
            else float("-inf")
        )

        second_score = (
            second.rerank_score
            if second.rerank_score is not None
            else float("-inf")
        )

        # -----------------------------------------------------
        # Higher rerank score
        # -----------------------------------------------------

        if second_score > first_score:

            return (
                second,
                first,
            )

        if first_score > second_score:

            return (
                first,
                second,
            )

        # -----------------------------------------------------
        # Same rerank score
        #
        # Prefer newer issuance date if available.
        # Dates in the current dataset are strings, so
        # lexical comparison works for ISO-like dates.
        # -----------------------------------------------------

        first_date = (
            first.issuance_date
            or ""
        )

        second_date = (
            second.issuance_date
            or ""
        )

        if second_date > first_date:

            return (
                second,
                first,
            )

        return (
            first,
            second,
        )

    # =========================================================
    # Text normalization
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Chuẩn hóa text trước khi so sánh duplicate.

        Không thay đổi dữ liệu gốc.
        """

        text = unicodedata.normalize(
            "NFKC",
            text,
        )

        text = text.casefold()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()