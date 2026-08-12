from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from vn_legal_rag.config import CHUNK_FILE


# ============================================================
# Config
# ============================================================

CONFLICT_FILE = Path(
    "data/chunk_duplicate_conflicts.csv"
)

# Số chunk_id conflict muốn kiểm tra
SAMPLE_SIZE = 20

# ============================================================
# Load conflict IDs
# ============================================================

def load_conflict_ids(
    path: Path,
    limit: int,
) -> list[str]:

    chunk_ids = []

    seen = set()

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            chunk_id = row["chunk_id"]

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            chunk_ids.append(chunk_id)

            if len(chunk_ids) >= limit:
                break

    return chunk_ids


# ============================================================
# Scan chunks
# ============================================================

def find_conflicts(
    chunk_file: Path,
    target_ids: set[str],
):
    """
    Stream chunks.jsonl.

    Không load toàn bộ file vào RAM.

    Chỉ lưu những record có chunk_id
    nằm trong target_ids.
    """

    records = defaultdict(list)

    found_ids = set()

    with open(
        chunk_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Scanning chunks",
            unit="chunk",
            dynamic_ncols=True,
        ):

            if not line.strip():
                continue

            data = json.loads(line)

            chunk_id = data.get(
                "chunk_id"
            )

            if chunk_id not in target_ids:
                continue

            records[chunk_id].append(data)

            found_ids.add(chunk_id)

    return records, found_ids


# ============================================================
# Print
# ============================================================

def print_record(
    record: dict,
    number: int,
):

    print()
    print(
        f"    Record #{number}"
    )
    print(
        "    " + "-" * 56
    )

    print(
        f"    chunk_id        : "
        f"{record.get('chunk_id')}"
    )

    print(
        f"    document_id     : "
        f"{record.get('document_id')}"
    )

    print(
        f"    article_no      : "
        f"{record.get('article_no')}"
    )

    print(
        f"    clause_no       : "
        f"{record.get('clause_no')}"
    )

    print(
        f"    point_no        : "
        f"{record.get('point_no')}"
    )

    print(
        f"    chunk_index     : "
        f"{record.get('chunk_index')}"
    )

    print(
        f"    sub_chunk_index : "
        f"{record.get('sub_chunk_index')}"
    )

    print(
        f"    text            :"
    )

    text = record.get(
        "text",
        "",
    )

    print(
        "    " + str(text).replace(
            "\n",
            "\n    ",
        )
    )


def print_conflicts(
    records: dict[str, list[dict]],
    chunk_ids: list[str],
):

    print()
    print("=" * 70)
    print("CONFLICT INSPECTION")
    print("=" * 70)

    for index, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):

        print()
        print(
            "#" * 70
        )

        print(
            f"CONFLICT {index}/{len(chunk_ids)}"
        )

        print(
            f"chunk_id = {chunk_id}"
        )

        print(
            "#" * 70
        )

        chunk_records = records.get(
            chunk_id,
            [],
        )

        if not chunk_records:

            print(
                "    NOT FOUND"
            )

            continue

        print(
            f"    Occurrences: "
            f"{len(chunk_records)}"
        )

        for number, record in enumerate(
            chunk_records,
            start=1,
        ):

            print_record(
                record,
                number,
            )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INSPECT CHUNK ID CONFLICTS")
    print("=" * 70)
    print()

    print(
        f"Chunk file    : {CHUNK_FILE}"
    )

    print(
        f"Conflict file : {CONFLICT_FILE}"
    )

    print(
        f"Sample size   : {SAMPLE_SIZE}"
    )

    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not CONFLICT_FILE.exists():

        raise FileNotFoundError(
            f"Conflict file not found: "
            f"{CONFLICT_FILE}"
        )

    if not Path(CHUNK_FILE).exists():

        raise FileNotFoundError(
            f"Chunk file not found: "
            f"{CHUNK_FILE}"
        )

    # --------------------------------------------------------
    # Load conflict IDs
    # --------------------------------------------------------

    print(
        "Loading conflict chunk_ids..."
    )

    chunk_ids = load_conflict_ids(
        CONFLICT_FILE,
        SAMPLE_SIZE,
    )

    if not chunk_ids:

        print(
            "No conflict chunk_ids found."
        )

        return

    print(
        f"Loaded {len(chunk_ids):,} "
        f"conflict chunk_ids."
    )

    print()

    for chunk_id in chunk_ids:

        print(
            f"  - {chunk_id}"
        )

    print()

    # --------------------------------------------------------
    # Scan JSONL
    # --------------------------------------------------------

    print(
        "Scanning chunks.jsonl..."
    )

    records, found_ids = find_conflicts(
        Path(CHUNK_FILE),
        set(chunk_ids),
    )

    print()

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_conflicts(
        records,
        chunk_ids,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print(
        f"Requested conflict IDs : "
        f"{len(chunk_ids):,}"
    )

    print(
        f"Found in chunks.jsonl   : "
        f"{len(found_ids):,}"
    )

    print(
        f"Not found               : "
        f"{len(chunk_ids - found_ids):,}"
    )

    total_records = sum(
        len(records[chunk_id])
        for chunk_id in found_ids
    )

    print(
        f"Matching records        : "
        f"{total_records:,}"
    )

    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()