import sqlite3
from pathlib import Path
import json

from vn_legal_rag.config import CHUNK_FILE


DB_FILE = Path("data/chunk_id_check.sqlite")


def main():

    print("=" * 60)
    print("CHECK UNIQUE CHUNK IDS")
    print("=" * 60)
    print()

    print(f"Chunk file : {CHUNK_FILE}")
    print(f"Database   : {DB_FILE}")
    print()

    DB_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(DB_FILE)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunk_ids (
            chunk_id TEXT PRIMARY KEY
        )
        """
    )

    conn.commit()

    total = 0
    duplicates = 0

    print("Scanning...")

    with open(
        CHUNK_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            data = json.loads(line)

            chunk_id = data["chunk_id"]

            total += 1

            cursor.execute(
                "INSERT OR IGNORE INTO chunk_ids (chunk_id) VALUES (?)",
                (chunk_id,),
            )

            if cursor.rowcount == 0:
                duplicates += 1

            if total % 100_000 == 0:

                conn.commit()

                print(
                    f"Processed: {total:,} | "
                    f"Unique: {total - duplicates:,} | "
                    f"Duplicates: {duplicates:,}"
                )

    conn.commit()

    cursor.execute(
        "SELECT COUNT(*) FROM chunk_ids"
    )

    unique_count = cursor.fetchone()[0]

    conn.close()

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print()

    print(f"Total chunks       : {total:,}")
    print(f"Unique chunk_ids   : {unique_count:,}")
    print(f"Duplicate entries  : {duplicates:,}")
    print()

    if total == unique_count:

        print("OK: All chunk_id are UNIQUE.")

    else:

        print(
            f"WARNING: {total - unique_count:,} "
            "duplicate chunk_id detected."
        )


if __name__ == "__main__":
    main()