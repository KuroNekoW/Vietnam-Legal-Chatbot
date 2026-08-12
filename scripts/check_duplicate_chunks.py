from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

from vn_legal_rag.config import CHUNK_FILE


# ============================================================
# Config
# ============================================================

DB_FILE = Path("data/chunk_duplicate_check.sqlite")

CONFLICT_FILE = Path(
    "data/chunk_duplicate_conflicts.csv"
)

BATCH_SIZE = 10_000

# Số conflict tối đa ghi ra CSV.
# Không giới hạn số lượng conflict trong database.
MAX_CONFLICT_SAMPLES = 1_000


# ============================================================
# Content hash
# ============================================================

def content_hash(data: dict) -> str:
    """
    Tạo hash cho toàn bộ chunk ngoại trừ chunk_id.

    Nếu cùng chunk_id xuất hiện nhiều lần:

        hash giống nhau
            -> duplicate hoàn toàn giống nhau

        hash khác nhau
            -> cùng chunk_id nhưng nội dung khác nhau
    """

    content = {
        key: value
        for key, value in data.items()
        if key != "chunk_id"
    }

    serialized = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================
# Database
# ============================================================

def create_database(conn: sqlite3.Connection):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            conflicting INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_chunks_conflicting
        ON chunks(conflicting)
        """
    )

    conn.commit()


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CHECK DUPLICATE CHUNKS")
    print("=" * 70)
    print()

    print(f"Chunk file : {CHUNK_FILE}")
    print(f"SQLite DB  : {DB_FILE}")
    print(f"Conflicts  : {CONFLICT_FILE}")
    print()

    DB_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFLICT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SQLite
    # --------------------------------------------------------

    print("Opening SQLite database...")

    conn = sqlite3.connect(
        DB_FILE,
    )

    # Tối ưu tốc độ ghi.
    #
    # WAL giúp SQLite ổn định hơn.
    #
    # synchronous=NORMAL giảm fsync nhưng vẫn khá an toàn.
    #

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA temp_store=FILE"
    )

    conn.execute(
        "PRAGMA cache_size=-65536"
    )
    # ~64 MB SQLite cache

    create_database(conn)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_rows = 0

    unique_ids = 0

    duplicate_rows = 0

    identical_duplicate_rows = 0

    conflicting_duplicate_rows = 0

    conflict_samples = 0

    # --------------------------------------------------------
    # Conflict CSV
    # --------------------------------------------------------

    conflict_file = open(
        CONFLICT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    )

    conflict_writer = csv.writer(
        conflict_file
    )

    conflict_writer.writerow(
        [
            "chunk_id",
            "first_hash",
            "duplicate_hash",
        ]
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    print("Scanning chunks...")
    print()

    batch = []

    def flush_batch():

        nonlocal unique_ids
        nonlocal duplicate_rows
        nonlocal identical_duplicate_rows
        nonlocal conflicting_duplicate_rows
        nonlocal conflict_samples

        if not batch:
            return

        for chunk_id, chunk_hash in batch:

            cursor = conn.execute(
                """
                SELECT
                    content_hash,
                    occurrences
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )

            row = cursor.fetchone()

            # ------------------------------------------------
            # New chunk_id
            # ------------------------------------------------

            if row is None:

                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        content_hash,
                        occurrences,
                        conflicting
                    )
                    VALUES (?, ?, 1, 0)
                    """,
                    (
                        chunk_id,
                        chunk_hash,
                    ),
                )

                unique_ids += 1

                continue

            # ------------------------------------------------
            # Duplicate chunk_id
            # ------------------------------------------------

            first_hash = row[0]

            occurrences = row[1]

            duplicate_rows += 1

            # ------------------------------------------------
            # Same content
            # ------------------------------------------------

            if first_hash == chunk_hash:

                identical_duplicate_rows += 1

                conn.execute(
                    """
                    UPDATE chunks
                    SET occurrences = ?
                    WHERE chunk_id = ?
                    """,
                    (
                        occurrences + 1,
                        chunk_id,
                    ),
                )

            # ------------------------------------------------
            # Same ID, different content
            # ------------------------------------------------

            else:

                conflicting_duplicate_rows += 1

                conn.execute(
                    """
                    UPDATE chunks
                    SET
                        occurrences = ?,
                        conflicting = 1
                    WHERE chunk_id = ?
                    """,
                    (
                        occurrences + 1,
                        chunk_id,
                    ),
                )

                if conflict_samples < MAX_CONFLICT_SAMPLES:

                    conflict_writer.writerow(
                        [
                            chunk_id,
                            first_hash,
                            chunk_hash,
                        ]
                    )

                    conflict_samples += 1

        batch.clear()

    # --------------------------------------------------------
    # Read JSONL
    # --------------------------------------------------------

    with open(
        CHUNK_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in tqdm(
            f,
            desc="Scanning",
            unit="chunk",
            dynamic_ncols=True,
        ):

            if not line.strip():
                continue

            data = json.loads(line)

            chunk_id = data.get(
                "chunk_id"
            )

            if chunk_id is None:

                continue

            chunk_hash = content_hash(
                data
            )

            batch.append(
                (
                    chunk_id,
                    chunk_hash,
                )
            )

            total_rows += 1

            if len(batch) >= BATCH_SIZE:

                flush_batch()

                conn.commit()

    # --------------------------------------------------------
    # Remaining
    # --------------------------------------------------------

    if batch:

        flush_batch()

        conn.commit()

    conflict_file.close()

    # --------------------------------------------------------
    # Final DB statistics
    # --------------------------------------------------------

    cursor = conn.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        """
    )

    unique_ids = cursor.fetchone()[0]

    cursor = conn.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        WHERE conflicting = 1
        """
    )

    conflicting_ids = cursor.fetchone()[0]

    cursor = conn.execute(
        """
        SELECT MAX(occurrences)
        FROM chunks
        """
    )

    max_occurrences = cursor.fetchone()[0] or 0

    conn.close()

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print()

    print(
        f"Total chunks             : "
        f"{total_rows:,}"
    )

    print(
        f"Unique chunk_ids         : "
        f"{unique_ids:,}"
    )

    print(
        f"Duplicate entries        : "
        f"{duplicate_rows:,}"
    )

    print(
        f"Identical duplicates     : "
        f"{identical_duplicate_rows:,}"
    )

    print(
        f"Conflicting duplicates   : "
        f"{conflicting_duplicate_rows:,}"
    )

    print(
        f"Conflicting chunk_ids    : "
        f"{conflicting_ids:,}"
    )

    print(
        f"Max occurrences / ID     : "
        f"{max_occurrences:,}"
    )

    print()

    # --------------------------------------------------------
    # Qdrant comparison
    # --------------------------------------------------------

    print(
        "Qdrant currently contains:"
    )

    print(
        "8,875,405 points "
        "(from your current collection)"
    )

    print()

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if conflicting_ids == 0:

        print(
            "RESULT: Duplicate chunk_ids have "
            "identical content."
        )

        print(
            "The duplicates are safe to "
            "deduplicate at the vector level."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            f"{conflicting_ids:,} chunk_ids "
            "have different content."
        )

        print(
            "These MUST be investigated because "
            "Qdrant uses chunk_id-derived point IDs."
        )

    print()

    print(
        f"Conflict samples written to:"
    )

    print(
        CONFLICT_FILE
    )

    print()

    print("=" * 70)
    print("CHECK FINISHED")
    print("=" * 70)
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()