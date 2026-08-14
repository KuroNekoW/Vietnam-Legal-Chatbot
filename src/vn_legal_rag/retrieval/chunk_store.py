from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable


class ChunkStore:
    """
    Persistent storage cho chunk text và metadata đầy đủ.

    Qdrant:
        vector + metadata nhỏ

    ChunkStore:
        text + metadata đầy đủ

    SQLite được sử dụng để lookup chunk theo chunk_id
    mà không phải scan toàn bộ chunks.jsonl mỗi query.
    """

    def __init__(
        self,
        database_path: str | Path,
    ):
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.database_path),
        )

        self.connection.execute(
            """
            PRAGMA journal_mode=WAL
            """
        )

        self.connection.execute(
            """
            PRAGMA synchronous=NORMAL
            """
        )

        self.connection.execute(
            """
            PRAGMA temp_store=MEMORY
            """
        )

        self._create_table()

    # ==========================================================
    # Database
    # ==========================================================

    def _create_table(self):

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (

                chunk_id TEXT PRIMARY KEY,

                document_id INTEGER,

                article TEXT,
                article_no INTEGER,

                clause TEXT,
                clause_no INTEGER,

                point TEXT,
                point_no TEXT,

                chunk_index INTEGER,
                sub_chunk_index INTEGER,

                start_char INTEGER,
                end_char INTEGER,

                title TEXT,

                legal_type TEXT,
                legal_sectors TEXT,

                issuing_authority TEXT,
                issuance_date TEXT,

                url TEXT,
                signers TEXT,

                text TEXT
            )
            """
        )

        self.connection.commit()

    # ==========================================================
    # Build database
    # ==========================================================

    def build_from_jsonl(
        self,
        jsonl_path: str | Path,
        batch_size: int = 5000,
        total: int | None = None,
    ):
        """
        Build SQLite chunk store từ chunks.jsonl.

        Parameters
        ----------
        jsonl_path:
            Đường dẫn tới chunks.jsonl.

        batch_size:
            Số chunk xử lý trong một transaction.

        total:
            Tổng số dòng/chunk trong JSONL.
            Dùng cho tqdm progress bar.
        """

        jsonl_path = Path(jsonl_path)

        batch = []

        from tqdm import tqdm

        progress = tqdm(
            total=total,
            desc="Building SQLite",
            unit="chunk",
            dynamic_ncols=True,
            colour="green",
        )

        try:

            with jsonl_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    chunk = json.loads(line)

                    batch.append(chunk)

                    if len(batch) >= batch_size:

                        self._insert_batch(
                            batch
                        )

                        progress.update(
                            len(batch)
                        )

                        batch.clear()

                # --------------------------------------------
                # Remaining batch
                # --------------------------------------------

                if batch:

                    self._insert_batch(
                        batch
                    )

                    progress.update(
                        len(batch)
                    )

                    batch.clear()

        finally:

            progress.close()

    def _insert_batch(
        self,
        chunks,
    ):

        rows = [

            (
                chunk["chunk_id"],
                chunk.get("document_id"),

                chunk.get("article"),
                chunk.get("article_no"),

                chunk.get("clause"),
                chunk.get("clause_no"),

                chunk.get("point"),
                chunk.get("point_no"),

                chunk.get("chunk_index"),
                chunk.get("sub_chunk_index", 0),

                chunk.get("start_char"),
                chunk.get("end_char"),

                chunk.get("title"),

                chunk.get("legal_type"),
                chunk.get("legal_sectors"),

                chunk.get("issuing_authority"),
                chunk.get("issuance_date"),

                chunk.get("url"),
                chunk.get("signers"),

                chunk.get("text"),
            )

            for chunk in chunks
        ]

        self.connection.executemany(
            """
            INSERT OR REPLACE INTO chunks (

                chunk_id,
                document_id,

                article,
                article_no,

                clause,
                clause_no,

                point,
                point_no,

                chunk_index,
                sub_chunk_index,

                start_char,
                end_char,

                title,

                legal_type,
                legal_sectors,

                issuing_authority,
                issuance_date,

                url,
                signers,

                text

            )

            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )

        self.connection.commit()

    # ==========================================================
    # Lookup
    # ==========================================================

    def get(
        self,
        chunk_id: str,
    ) -> dict | None:

        cursor = self.connection.execute(
            """
            SELECT
                chunk_id,
                document_id,

                article,
                article_no,

                clause,
                clause_no,

                point,
                point_no,

                chunk_index,
                sub_chunk_index,

                start_char,
                end_char,

                title,

                legal_type,
                legal_sectors,

                issuing_authority,
                issuance_date,

                url,
                signers,

                text

            FROM chunks

            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        columns = [
            "chunk_id",
            "document_id",

            "article",
            "article_no",

            "clause",
            "clause_no",

            "point",
            "point_no",

            "chunk_index",
            "sub_chunk_index",

            "start_char",
            "end_char",

            "title",

            "legal_type",
            "legal_sectors",

            "issuing_authority",
            "issuance_date",

            "url",
            "signers",

            "text",
        ]

        return dict(
            zip(columns, row)
        )

    def get_many(
        self,
        chunk_ids: Iterable[str],
    ) -> dict[str, dict]:

        chunk_ids = list(chunk_ids)

        if not chunk_ids:
            return {}

        result = {}

        # SQLite có giới hạn số parameter tùy version.
        # Chia nhỏ để an toàn.

        batch_size = 500

        columns = [
            "chunk_id",
            "document_id",

            "article",
            "article_no",

            "clause",
            "clause_no",

            "point",
            "point_no",

            "chunk_index",
            "sub_chunk_index",

            "start_char",
            "end_char",

            "title",

            "legal_type",
            "legal_sectors",

            "issuing_authority",
            "issuance_date",

            "url",
            "signers",

            "text",
        ]

        for start in range(
            0,
            len(chunk_ids),
            batch_size,
        ):

            batch = chunk_ids[
                start:start + batch_size
            ]

            placeholders = ",".join(
                "?"
                for _ in batch
            )

            cursor = self.connection.execute(
                f"""
                SELECT
                    chunk_id,
                    document_id,

                    article,
                    article_no,

                    clause,
                    clause_no,

                    point,
                    point_no,

                    chunk_index,
                    sub_chunk_index,

                    start_char,
                    end_char,

                    title,

                    legal_type,
                    legal_sectors,

                    issuing_authority,
                    issuance_date,

                    url,
                    signers,

                    text

                FROM chunks

                WHERE chunk_id IN ({placeholders})
                """,
                batch,
            )

            for row in cursor.fetchall():

                record = dict(
                    zip(
                        columns,
                        row,
                    )
                )

                result[
                    record["chunk_id"]
                ] = record

        return result

    # ==========================================================
    # Statistics
    # ==========================================================

    @property
    def count(self) -> int:

        return self.connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        ).fetchone()[0]

    # ==========================================================
    # Close
    # ==========================================================

    def close(self):

        self.connection.close()

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()