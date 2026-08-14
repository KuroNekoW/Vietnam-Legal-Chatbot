from __future__ import annotations

import json

from vn_legal_rag.llm import LocalLLM

from .prompt import build_normalization_prompt
from .schema import NormalizedQuery


class QueryNormalizer:
    """
    Normalize user legal queries using a local LLM.
    """

    def __init__(
        self,
        llm: LocalLLM,
    ):

        self.llm = llm

    # ========================================================
    # Normalize
    # ========================================================

    def normalize(
        self,
        query: str,
    ) -> NormalizedQuery:

        query = query.strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        prompt = build_normalization_prompt(
            query
        )

        #
        # Pydantic -> JSON Schema
        #

        schema = (
            NormalizedQuery
            .model_json_schema()
        )

        response = self.llm.generate(
            prompt=prompt,
            response_schema=schema,
            temperature=0.0,
            max_tokens=512,
        )

        #
        # Parse JSON
        #

        data = self._parse_json(
            response
        )

        #
        # Validate exact schema
        #

        return NormalizedQuery.model_validate(
            data
        )

    # ========================================================
    # JSON parser
    # ========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        text = text.strip()

        #
        # Code fence fallback
        #

        if text.startswith("```"):

            lines = text.splitlines()

            lines = [
                line
                for line in lines
                if not line.strip().startswith(
                    "```"
                )
            ]

            text = "\n".join(
                lines
            ).strip()

        #
        # Direct JSON
        #

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:

            pass

        #
        # Find JSON object
        #

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                "LLM did not return valid JSON.\n"
                f"Output:\n{text}"
            )

        try:

            return json.loads(
                text[start:end + 1]
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Failed to parse LLM JSON output.\n"
                f"Output:\n{text}"
            ) from exc