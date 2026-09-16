from __future__ import annotations

from typing import Any

from vn_legal_rag.context.models import ContextResult
from vn_legal_rag.generation.models import LegalAnswer
from vn_legal_rag.generation.prompt import (
    build_system_prompt,
    build_user_prompt,
)


class AnswerGenerator:
    """
    Generate structured legal answers using the local LLM.

    Responsibilities:
        - build system/user prompts;
        - call LocalLLM;
        - validate the returned JSON against LegalAnswer.

    The generator does not perform retrieval or context selection.
    """

    def __init__(
        self,
        llm: Any,
        *,
        max_tokens: int = 768,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        *,
        query: str,
        context: ContextResult | str,
    ) -> LegalAnswer:
        """
        Generate a structured answer from a query and selected context.
        """
        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        context_text = self._prepare_context(
            context
        )

        if not context_text:
            return LegalAnswer(
                answer=(
                    "Không có đủ thông tin pháp lý trong "
                    "context được cung cấp để trả lời câu hỏi."
                ),
                citations=[],
                insufficient_evidence=True,
                note=(
                    "Context đầu vào không chứa nội dung pháp lý "
                    "để làm căn cứ trả lời."
                ),
            )

        system_prompt = build_system_prompt()

        user_prompt = build_user_prompt(
            query=query,
            context=context_text,
        )

        response_schema = (
            LegalAnswer.model_json_schema()
        )

        raw_response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._validate_response(
            raw_response
        )

    @staticmethod
    def _prepare_context(
        context: ContextResult | str,
    ) -> str:
        if isinstance(
            context,
            ContextResult,
        ):
            return context.rendered_text.strip()

        if isinstance(context, str):
            return context.strip()

        raise TypeError(
            "context must be ContextResult or str"
        )

    @staticmethod
    def _validate_response(
        raw_response: Any,
    ) -> LegalAnswer:
        """
        Convert the LocalLLM response into LegalAnswer.

        LocalLLM normally returns a dict after JSON parsing, but this
        also accepts a JSON string for robustness.
        """
        if isinstance(
            raw_response,
            LegalAnswer,
        ):
            return raw_response

        if isinstance(
            raw_response,
            dict,
        ):
            return LegalAnswer.model_validate(
                raw_response
            )

        if isinstance(
            raw_response,
            str,
        ):
            import json

            try:
                data = json.loads(
                    raw_response
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "LLM returned invalid JSON."
                ) from exc

            return LegalAnswer.model_validate(
                data
            )

        raise TypeError(
            "Unsupported LLM response type: "
            f"{type(raw_response).__name__}"
        )


__all__ = [
    "AnswerGenerator",
]