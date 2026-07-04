import json
from pathlib import Path
from typing import Generator, Iterable

from vn_legal_rag.models import LegalDocument
from vn_legal_rag.models import Chunk


def save_jsonl(
    documents: Iterable[LegalDocument],
    output_path: Path,
) -> None:
    """
    Save LegalDocument objects to a JSONL file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for document in documents:

            json.dump(
                document.model_dump(),
                f,
                ensure_ascii=False,
            )

            f.write("\n")


def load_jsonl(
    input_path: Path,
) -> Generator[LegalDocument, None, None]:
    """
    Lazily load LegalDocument objects from a JSONL file.
    """

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            yield LegalDocument.model_validate_json(
                line
            )

# Chunks

def save_chunks_jsonl(
    chunks: Iterable[Chunk],
    output_path: Path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:

        for chunk in chunks:

            json.dump(
                chunk.model_dump(),
                f,
                ensure_ascii=False,
            )

            f.write("\n")


def load_chunks_jsonl(
    input_path: Path,
):

    with open(input_path, encoding="utf-8") as f:

        for line in f:

            yield Chunk.model_validate_json(line)