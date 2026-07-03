import json
from pathlib import Path
from typing import Iterable

from vn_legal_rag.models import LegalDocument


def save_jsonl(
    documents: Iterable[LegalDocument],
    output_path: Path,
):

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
):

    with open(
        input_path,
        encoding="utf-8",
    ) as f:

        for line in f:

            yield LegalDocument.model_validate_json(
                line
            )