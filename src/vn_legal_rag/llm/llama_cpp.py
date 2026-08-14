from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_cpp import Llama


class LocalLLM:
    """
    Local LLM backend using llama.cpp.

    Designed for:
    - Qwen3 GGUF
    - Offline inference
    - CUDA GPU offloading
    - JSON schema constrained output
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found: {self.model_path}"
            )

        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(
            f"Loading local LLM: {self.model_path}"
        )

        self.model = Llama(
            model_path=str(self.model_path),

            n_ctx=n_ctx,

            # RTX 3070 / 4060:
            # offload all layers that fit on GPU.
            n_gpu_layers=n_gpu_layers,

            n_threads=n_threads,

            verbose=False,
        )

        print("Local LLM loaded.")

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        prompt: str,
        response_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:

        if max_tokens is None:
            max_tokens = self.max_tokens

        if temperature is None:
            temperature = self.temperature

        #
        # Qwen3 soft switch:
        # disable thinking mode.
        #

        if "/no_think" not in prompt:
            prompt = (
                prompt.rstrip()
                + "\n\n/no_think"
            )

        #
        # Base response format.
        #

        response_format: dict[str, Any] = {
            "type": "json_object",
        }

        #
        # Add JSON Schema when provided.
        #

        if response_schema is not None:

            response_format["schema"] = (
                response_schema
            )

        response = self.model.create_chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là một trợ lý xử lý "
                        "truy vấn pháp luật Việt Nam."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=temperature,

            top_p=0.9,

            max_tokens=max_tokens,

            response_format=response_format,
        )

        content = (
            response["choices"][0]
            ["message"]
            ["content"]
        )

        if not content:
            raise RuntimeError(
                "Local LLM returned empty content."
            )

        return content.strip()