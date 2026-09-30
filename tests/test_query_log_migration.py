import sqlite3
from pathlib import Path

from company_agent.database import KnowledgeRepository


def test_existing_query_log_rows_receive_stage_metric_defaults(tmp_path: Path) -> None:
    database_path = tmp_path / "knowledge.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE query_logs (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                source_ids TEXT NOT NULL,
                retrieval_ms INTEGER NOT NULL,
                generation_ms INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO query_logs VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old-request", "hashed-question", "grounded", "[]", 12, 34, "2026-09-21"),
        )

    repository = KnowledgeRepository(database_path)
    repository.initialize()

    with repository.connect() as connection:
        row = connection.execute(
            """
            SELECT source_snapshots, model_name, prompt_version, topic_check_ms,
                   initial_generation_ms, repair_ms, repair_attempts, queue_ms,
                   total_ms, cache_hit
            FROM query_logs WHERE id = ?
            """,
            ("old-request",),
        ).fetchone()
    assert row is not None
    assert tuple(row) == ("[]", "", "", 0, 0, 0, 0, 0, 0, 0)
