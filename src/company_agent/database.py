import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PUBLIC_OWNER_COMPANY_ID = "wuxi-gougelila"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def pack_embedding(values: list[float]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def unpack_embedding(value: bytes) -> np.ndarray:
    return np.frombuffer(value, dtype=np.float32)


def _document_dict(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    raw_product_ids = value.get("product_ids") or "[]"
    try:
        value["product_ids"] = json.loads(raw_product_ids)
    except (TypeError, json.JSONDecodeError):
        value["product_ids"] = []
    value["is_withdrawn"] = bool(value.get("is_withdrawn", 0))
    return value


def _media_dict(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    raw_product_ids = value.get("product_ids") or "[]"
    try:
        value["product_ids"] = json.loads(raw_product_ids)
    except (TypeError, json.JSONDecodeError):
        value["product_ids"] = []
    value["enabled"] = bool(value.get("enabled", 0))
    value["extraction_status"] = value.get("extraction_status") or "current"
    return value


class KnowledgeRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _add_missing_columns(
        connection: sqlite3.Connection,
        table: str,
        definitions: dict[str, str],
    ) -> set[str]:
        present = {
            str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        added: set[str] = set()
        for name, definition in definitions.items():
            if name in present:
                continue
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            added.add(name)
        return added

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    knowledge_domain TEXT NOT NULL DEFAULT 'company_public',
                    owner_company_id TEXT NOT NULL DEFAULT 'wuxi-gougelila',
                    brand_id TEXT,
                    product_ids TEXT NOT NULL DEFAULT '[]',
                    visibility TEXT NOT NULL DEFAULT 'public',
                    publication_status TEXT NOT NULL DEFAULT 'published',
                    review_status TEXT NOT NULL DEFAULT 'approved',
                    reviewer_id TEXT,
                    reviewed_at TEXT,
                    effective_from TEXT,
                    effective_to TEXT,
                    source_version INTEGER NOT NULL DEFAULT 1,
                    active_version INTEGER,
                    is_withdrawn INTEGER NOT NULL DEFAULT 0,
                    evidence_kind TEXT NOT NULL DEFAULT 'official_material',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    sheet TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    embedding BLOB NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, position)
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    retrieval_ms INTEGER NOT NULL,
                    generation_ms INTEGER NOT NULL,
                    reason_code TEXT NOT NULL DEFAULT 'LEGACY',
                    policy_version TEXT NOT NULL DEFAULT 'legacy',
                    evidence_status TEXT NOT NULL DEFAULT 'missing',
                    fallback_kind TEXT,
                    source_snapshots TEXT NOT NULL DEFAULT '[]',
                    model_name TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    topic_check_ms INTEGER NOT NULL DEFAULT 0,
                    initial_generation_ms INTEGER NOT NULL DEFAULT 0,
                    repair_ms INTEGER NOT NULL DEFAULT 0,
                    repair_attempts INTEGER NOT NULL DEFAULT 0,
                    repair_reason TEXT,
                    queue_ms INTEGER NOT NULL DEFAULT 0,
                    total_ms INTEGER NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS media_assets (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    image_key TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    page INTEGER,
                    sheet TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    cell_ref TEXT,
                    caption TEXT NOT NULL DEFAULT '',
                    alt_text TEXT NOT NULL DEFAULT '',
                    product_ids TEXT NOT NULL DEFAULT '[]',
                    search_text TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    review_status TEXT NOT NULL DEFAULT 'approved',
                    extraction_status TEXT NOT NULL DEFAULT 'current',
                    extracted_caption TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(document_id, image_key)
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    knowledge_domain TEXT NOT NULL,
                    owner_company_id TEXT NOT NULL,
                    brand_id TEXT,
                    product_ids TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL,
                    effective_from TEXT,
                    effective_to TEXT,
                    reviewer_id TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    PRIMARY KEY(document_id, version)
                );

                CREATE TABLE IF NOT EXISTS chunk_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    sheet TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    embedding BLOB NOT NULL,
                    enabled INTEGER NOT NULL,
                    PRIMARY KEY(document_id, version, chunk_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS media_asset_versions (
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    asset_id TEXT NOT NULL,
                    image_key TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    source_kind TEXT NOT NULL,
                    page INTEGER,
                    sheet TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    cell_ref TEXT,
                    caption TEXT NOT NULL,
                    alt_text TEXT NOT NULL,
                    product_ids TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    PRIMARY KEY(document_id, version, asset_id),
                    FOREIGN KEY(document_id, version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS publication_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    version INTEGER,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            document_columns = self._add_missing_columns(
                connection,
                "documents",
                {
                    "knowledge_domain": "TEXT NOT NULL DEFAULT 'company_public'",
                    "owner_company_id": "TEXT NOT NULL DEFAULT 'wuxi-gougelila'",
                    "brand_id": "TEXT",
                    "product_ids": "TEXT NOT NULL DEFAULT '[]'",
                    "visibility": "TEXT NOT NULL DEFAULT 'public'",
                    "publication_status": "TEXT NOT NULL DEFAULT 'published'",
                    "review_status": "TEXT NOT NULL DEFAULT 'approved'",
                    "reviewer_id": "TEXT",
                    "reviewed_at": "TEXT",
                    "effective_from": "TEXT",
                    "effective_to": "TEXT",
                    "source_version": "INTEGER NOT NULL DEFAULT 1",
                    "active_version": "INTEGER",
                    "is_withdrawn": "INTEGER NOT NULL DEFAULT 0",
                    "evidence_kind": "TEXT NOT NULL DEFAULT 'official_material'",
                },
            )
            self._add_missing_columns(
                connection,
                "media_assets",
                {
                    "extraction_status": "TEXT NOT NULL DEFAULT 'current'",
                    "extracted_caption": "TEXT NOT NULL DEFAULT ''",
                },
            )
            self._add_missing_columns(
                connection,
                "query_logs",
                {
                    "reason_code": "TEXT NOT NULL DEFAULT 'LEGACY'",
                    "policy_version": "TEXT NOT NULL DEFAULT 'legacy'",
                    "evidence_status": "TEXT NOT NULL DEFAULT 'missing'",
                    "fallback_kind": "TEXT",
                    "source_snapshots": "TEXT NOT NULL DEFAULT '[]'",
                    "model_name": "TEXT NOT NULL DEFAULT ''",
                    "prompt_version": "TEXT NOT NULL DEFAULT ''",
                    "topic_check_ms": "INTEGER NOT NULL DEFAULT 0",
                    "initial_generation_ms": "INTEGER NOT NULL DEFAULT 0",
                    "repair_ms": "INTEGER NOT NULL DEFAULT 0",
                    "repair_attempts": "INTEGER NOT NULL DEFAULT 0",
                    "repair_reason": "TEXT",
                    "queue_ms": "INTEGER NOT NULL DEFAULT 0",
                    "total_ms": "INTEGER NOT NULL DEFAULT 0",
                    "cache_hit": "INTEGER NOT NULL DEFAULT 0",
                },
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at)"
            )
            if "knowledge_domain" in document_columns:
                connection.execute(
                    """
                    UPDATE documents
                    SET knowledge_domain = 'product_public'
                    WHERE lower(title || ' ' || filename) LIKE '%ckin%'
                       OR (title || ' ' || filename) LIKE '%宠沁%'
                       OR (title || ' ' || filename) LIKE '%宣传册%'
                    """
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET reviewed_at = COALESCE(reviewed_at, updated_at), reviewer_id = 'migration'
                    WHERE review_status = 'approved'
                    """
                )
            connection.execute(
                "UPDATE documents SET is_withdrawn = 1 WHERE publication_status = 'withdrawn'"
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO document_versions (
                    document_id, version, title, knowledge_domain, owner_company_id,
                    brand_id, product_ids, visibility, evidence_kind, effective_from,
                    effective_to, reviewer_id, reviewed_at, published_at
                )
                SELECT id, source_version, title, knowledge_domain, owner_company_id,
                       brand_id, product_ids, visibility, evidence_kind, effective_from,
                       effective_to, COALESCE(reviewer_id, 'migration'),
                       COALESCE(reviewed_at, updated_at), updated_at
                FROM documents
                WHERE publication_status = 'published'
                  AND review_status = 'approved'
                  AND active_version IS NULL
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO chunk_versions (
                    document_id, version, chunk_id, position, content, page, sheet,
                    row_start, row_end, embedding, enabled
                )
                SELECT c.document_id, d.source_version, c.id, c.position, c.content,
                       c.page, c.sheet, c.row_start, c.row_end, c.embedding, c.enabled
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.source_version
                WHERE d.publication_status = 'published'
                  AND d.review_status = 'approved'
                  AND d.active_version IS NULL
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO media_asset_versions (
                    document_id, version, asset_id, image_key, filename, stored_path,
                    mime_type, sha256, width, height, source_kind, page, sheet,
                    row_start, row_end, cell_ref, caption, alt_text, product_ids,
                    search_text, enabled
                )
                SELECT m.document_id, d.source_version, m.id, m.image_key, m.filename,
                       m.stored_path, m.mime_type, m.sha256, m.width, m.height,
                       m.source_kind, m.page, m.sheet, m.row_start, m.row_end,
                       m.cell_ref, m.caption, m.alt_text, m.product_ids, m.search_text,
                       m.enabled
                FROM media_assets m
                JOIN documents d ON d.id = m.document_id
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.source_version
                WHERE d.publication_status = 'published'
                  AND d.review_status = 'approved'
                  AND d.active_version IS NULL
                  AND m.review_status = 'approved'
                """
            )
            connection.execute(
                """
                UPDATE documents
                SET active_version = source_version
                WHERE publication_status = 'published'
                  AND review_status = 'approved'
                  AND active_version IS NULL
                """
            )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id, position);
                CREATE INDEX IF NOT EXISTS idx_chunks_enabled
                    ON chunks(enabled);
                CREATE INDEX IF NOT EXISTS idx_documents_publication
                    ON documents(publication_status, review_status, visibility, knowledge_domain);
                CREATE INDEX IF NOT EXISTS idx_media_document
                    ON media_assets(document_id, enabled, review_status);
                CREATE INDEX IF NOT EXISTS idx_media_location
                    ON media_assets(document_id, page, sheet, row_start, row_end);
                CREATE INDEX IF NOT EXISTS idx_document_versions_public
                    ON document_versions(owner_company_id, visibility, knowledge_domain);
                CREATE INDEX IF NOT EXISTS idx_chunk_versions_active
                    ON chunk_versions(document_id, version, enabled, position);
                CREATE INDEX IF NOT EXISTS idx_media_versions_active
                    ON media_asset_versions(document_id, version, enabled);
                CREATE INDEX IF NOT EXISTS idx_publication_events_document
                    ON publication_events(document_id, created_at DESC);
                """
            )

    def create_document(self, values: dict[str, Any]) -> None:
        normalized = {
            "error_message": None,
            "knowledge_domain": "company_public",
            "owner_company_id": "wuxi-gougelila",
            "brand_id": None,
            "product_ids": "[]",
            "visibility": "public",
            "publication_status": "draft",
            "review_status": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "effective_from": None,
            "effective_to": None,
            "source_version": 1,
            "active_version": None,
            "is_withdrawn": 0,
            "evidence_kind": "official_material",
            **values,
        }
        if not isinstance(normalized["product_ids"], str):
            normalized["product_ids"] = json.dumps(normalized["product_ids"], ensure_ascii=False)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, title, filename, stored_path, file_type, size_bytes, sha256,
                    status, error_message, knowledge_domain, owner_company_id, brand_id,
                    product_ids, visibility, publication_status, review_status, reviewer_id,
                    reviewed_at, effective_from, effective_to, source_version, evidence_kind,
                    active_version, is_withdrawn, created_at, updated_at
                ) VALUES (
                    :id, :title, :filename, :stored_path, :file_type, :size_bytes, :sha256,
                    :status, :error_message, :knowledge_domain, :owner_company_id, :brand_id,
                    :product_ids, :visibility, :publication_status, :review_status, :reviewer_id,
                    :reviewed_at, :effective_from, :effective_to, :source_version, :evidence_kind,
                    :active_version, :is_withdrawn, :created_at, :updated_at
                )
                """,
                normalized,
            )

    def update_document_status(
        self, document_id: str, status: str, error_message: str | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error_message, utc_now(), document_id),
            )

    def update_document_metadata(self, document_id: str, values: dict[str, Any]) -> None:
        allowed = {
            "title",
            "knowledge_domain",
            "owner_company_id",
            "brand_id",
            "product_ids",
            "visibility",
            "publication_status",
            "review_status",
            "reviewer_id",
            "reviewed_at",
            "effective_from",
            "effective_to",
            "evidence_kind",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "product_ids" in updates:
            updates["product_ids"] = json.dumps(updates["product_ids"], ensure_ascii=False)
        if not updates:
            return
        assignments = [f"{name} = ?" for name in updates]
        parameters = list(updates.values())
        assignments.append("updated_at = ?")
        parameters.extend([utc_now(), document_id])
        with self.connect() as connection:
            connection.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                parameters,
            )

    @staticmethod
    def _record_publication_event(
        connection: sqlite3.Connection,
        document_id: str,
        version: int | None,
        event_type: str,
        actor_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO publication_events (
                document_id, version, event_type, actor_id, details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                version,
                event_type,
                actor_id,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    @classmethod
    def _mark_document_draft_in_transaction(
        cls,
        connection: sqlite3.Connection,
        document_id: str,
        actor_id: str,
        details: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        connection.execute(
            """
            UPDATE documents
            SET source_version = source_version + 1,
                publication_status = CASE
                    WHEN is_withdrawn = 1 THEN 'withdrawn' ELSE 'draft'
                END,
                review_status = 'pending',
                reviewer_id = NULL,
                reviewed_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (now, document_id),
        )
        row = connection.execute(
            "SELECT source_version FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise KeyError(document_id)
        version = int(row["source_version"])
        cls._record_publication_event(
            connection,
            document_id,
            version,
            "draft_changed",
            actor_id,
            details,
        )
        return version

    def update_document_draft(
        self,
        document_id: str,
        values: dict[str, Any],
        *,
        actor_id: str = "local-admin",
    ) -> None:
        allowed = {
            "title",
            "knowledge_domain",
            "owner_company_id",
            "brand_id",
            "product_ids",
            "visibility",
            "effective_from",
            "effective_to",
            "evidence_kind",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "product_ids" in updates:
            updates["product_ids"] = json.dumps(updates["product_ids"], ensure_ascii=False)
        if not updates:
            return
        assignments = [f"{name} = ?" for name in updates]
        parameters = list(updates.values())
        parameters.append(document_id)
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(document_id)
            self._mark_document_draft_in_transaction(
                connection,
                document_id,
                actor_id,
                {"fields": sorted(updates)},
            )

    def publish_document(
        self,
        document_id: str,
        *,
        reviewer_id: str,
        immutable_media_paths: dict[str, str] | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> int:
        immutable_media_paths = immutable_media_paths or {}
        metadata_updates = metadata_updates or {}
        now = utc_now()
        with self.connect() as connection:
            document = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                raise KeyError(document_id)
            allowed = {
                "title",
                "knowledge_domain",
                "owner_company_id",
                "brand_id",
                "product_ids",
                "visibility",
                "effective_from",
                "effective_to",
                "evidence_kind",
            }
            updates = {key: value for key, value in metadata_updates.items() if key in allowed}
            if "product_ids" in updates and not isinstance(updates["product_ids"], str):
                updates["product_ids"] = json.dumps(updates["product_ids"], ensure_ascii=False)
            changed_updates = {
                key: value for key, value in updates.items() if document[key] != value
            }
            if changed_updates:
                assignments = [f"{name} = ?" for name in changed_updates]
                connection.execute(
                    f"UPDATE documents SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                    [*changed_updates.values(), document_id],
                )
                self._mark_document_draft_in_transaction(
                    connection,
                    document_id,
                    reviewer_id,
                    {"fields": sorted(changed_updates), "publish_transaction": True},
                )
                document = connection.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if document is None:
                    raise KeyError(document_id)
            version = int(document["source_version"])
            existing = connection.execute(
                """
                SELECT 1 FROM document_versions
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO document_versions (
                        document_id, version, title, knowledge_domain, owner_company_id,
                        brand_id, product_ids, visibility, evidence_kind, effective_from,
                        effective_to, reviewer_id, reviewed_at, published_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        version,
                        document["title"],
                        document["knowledge_domain"],
                        document["owner_company_id"],
                        document["brand_id"],
                        document["product_ids"],
                        document["visibility"],
                        document["evidence_kind"],
                        document["effective_from"],
                        document["effective_to"],
                        reviewer_id,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunk_versions (
                        document_id, version, chunk_id, position, content, page, sheet,
                        row_start, row_end, embedding, enabled
                    )
                    SELECT document_id, ?, id, position, content, page, sheet,
                           row_start, row_end, embedding, enabled
                    FROM chunks
                    WHERE document_id = ?
                    """,
                    (version, document_id),
                )
                media_rows = connection.execute(
                    "SELECT * FROM media_assets WHERE document_id = ?", (document_id,)
                ).fetchall()
                connection.executemany(
                    """
                    INSERT INTO media_asset_versions (
                        document_id, version, asset_id, image_key, filename, stored_path,
                        mime_type, sha256, width, height, source_kind, page, sheet,
                        row_start, row_end, cell_ref, caption, alt_text, product_ids,
                        search_text, enabled
                    ) VALUES (
                        :document_id, :version, :asset_id, :image_key, :filename,
                        :stored_path, :mime_type, :sha256, :width, :height, :source_kind,
                        :page, :sheet, :row_start, :row_end, :cell_ref, :caption,
                        :alt_text, :product_ids, :search_text, :enabled
                    )
                    """,
                    [
                        {
                            "document_id": document_id,
                            "version": version,
                            "asset_id": row["id"],
                            "image_key": row["image_key"],
                            "filename": row["filename"],
                            "stored_path": immutable_media_paths.get(
                                str(row["id"]), str(row["stored_path"])
                            ),
                            "mime_type": row["mime_type"],
                            "sha256": row["sha256"],
                            "width": row["width"],
                            "height": row["height"],
                            "source_kind": row["source_kind"],
                            "page": row["page"],
                            "sheet": row["sheet"],
                            "row_start": row["row_start"],
                            "row_end": row["row_end"],
                            "cell_ref": row["cell_ref"],
                            "caption": row["caption"],
                            "alt_text": row["alt_text"],
                            "product_ids": row["product_ids"],
                            "search_text": row["search_text"],
                            "enabled": row["enabled"],
                        }
                        for row in media_rows
                    ],
                )
            connection.execute(
                """
                UPDATE documents
                SET active_version = ?, is_withdrawn = 0,
                    publication_status = 'published', review_status = 'approved',
                    reviewer_id = ?, reviewed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (version, reviewer_id, now, now, document_id),
            )
            connection.execute(
                """
                UPDATE media_assets
                SET review_status = 'approved', updated_at = ?
                WHERE document_id = ?
                """,
                (now, document_id),
            )
            self._record_publication_event(
                connection,
                document_id,
                version,
                "published",
                reviewer_id,
                {"reused_snapshot": existing is not None},
            )
        return version

    def withdraw_document(self, document_id: str, *, actor_id: str = "local-admin") -> None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT active_version FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise KeyError(document_id)
            connection.execute(
                """
                UPDATE documents
                SET is_withdrawn = 1, publication_status = 'withdrawn', updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), document_id),
            )
            self._record_publication_event(
                connection,
                document_id,
                int(row["active_version"]) if row["active_version"] is not None else None,
                "withdrawn",
                actor_id,
            )

    def rollback_document(
        self,
        document_id: str,
        version: int,
        *,
        actor_id: str = "local-admin",
    ) -> None:
        with self.connect() as connection:
            document = connection.execute(
                "SELECT source_version FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if document is None:
                raise KeyError(document_id)
            snapshot = connection.execute(
                """
                SELECT 1 FROM document_versions
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            ).fetchone()
            if snapshot is None:
                raise ValueError("指定发布版本不存在")
            has_newer_draft = int(document["source_version"]) != version
            connection.execute(
                """
                UPDATE documents
                SET active_version = ?, is_withdrawn = 0,
                    publication_status = ?, review_status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    version,
                    "draft" if has_newer_draft else "published",
                    "pending" if has_newer_draft else "approved",
                    utc_now(),
                    document_id,
                ),
            )
            self._record_publication_event(
                connection,
                document_id,
                version,
                "rolled_back",
                actor_id,
                {"draft_preserved": has_newer_draft},
            )

    def mark_document_content_changed(self, document_id: str) -> None:
        with self.connect() as connection:
            self._mark_document_draft_in_transaction(
                connection,
                document_id,
                "system",
                {"fields": ["content"]},
            )

    def find_document_by_sha256(self, sha256: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return _document_dict(row) if row else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = ?
                GROUP BY d.id
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        value = _document_dict(row)
        with self.connect() as connection:
            value["media_count"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM media_assets WHERE document_id = ?", (document_id,)
                ).fetchone()[0]
            )
        return value

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        documents = [_document_dict(row) for row in rows]
        with self.connect() as connection:
            media_counts = {
                str(row["document_id"]): int(row["media_count"])
                for row in connection.execute(
                    """
                    SELECT document_id, COUNT(*) AS media_count
                    FROM media_assets
                    GROUP BY document_id
                    """
                ).fetchall()
            }
        for document in documents:
            document["media_count"] = media_counts.get(str(document["id"]), 0)
        return documents

    def replace_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, position, content, page, sheet, row_start, row_end,
                    embedding, enabled, created_at, updated_at
                ) VALUES (
                    :id, :document_id, :position, :content, :page, :sheet, :row_start,
                    :row_end, :embedding, :enabled, :created_at, :updated_at
                )
                """,
                [
                    {
                        **chunk,
                        "document_id": document_id,
                        "enabled": int(chunk.get("enabled", True)),
                        "created_at": now,
                        "updated_at": now,
                    }
                    for chunk in chunks
                ],
            )

    def list_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, position, content, page, sheet, row_start,
                       row_end, enabled, created_at, updated_at
                FROM chunks
                WHERE document_id = ?
                ORDER BY position
                """,
                (document_id,),
            ).fetchall()
        return [{**dict(row), "enabled": bool(row["enabled"])} for row in rows]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def update_chunk(
        self,
        chunk_id: str,
        *,
        content: str | None = None,
        embedding: list[float] | None = None,
        enabled: bool | None = None,
    ) -> None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if content is not None:
            assignments.append("content = ?")
            values.append(content)
        if embedding is not None:
            assignments.append("embedding = ?")
            values.append(pack_embedding(embedding))
        if enabled is not None:
            assignments.append("enabled = ?")
            values.append(int(enabled))
        values.append(chunk_id)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE chunks SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                values,
            )

    def update_chunk_draft(
        self,
        chunk_id: str,
        *,
        content: str | None = None,
        embedding: list[float] | None = None,
        enabled: bool | None = None,
        actor_id: str = "local-admin",
    ) -> None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if content is not None:
            assignments.append("content = ?")
            values.append(content)
        if embedding is not None:
            assignments.append("embedding = ?")
            values.append(pack_embedding(embedding))
        if enabled is not None:
            assignments.append("enabled = ?")
            values.append(int(enabled))
        values.append(chunk_id)
        with self.connect() as connection:
            chunk = connection.execute(
                "SELECT document_id FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if chunk is None:
                raise KeyError(chunk_id)
            connection.execute(
                f"UPDATE chunks SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                values,
            )
            self._mark_document_draft_in_transaction(
                connection,
                str(chunk["document_id"]),
                actor_id,
                {
                    "chunk_id": chunk_id,
                    "fields": [
                        name
                        for name, present in (
                            ("content", content is not None),
                            ("enabled", enabled is not None),
                        )
                        if present
                    ],
                },
            )

    def update_chunk_embeddings(self, rows: Sequence[tuple[str, list[float]]]) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.executemany(
                "UPDATE chunks SET embedding = ?, updated_at = ? WHERE id = ?",
                [(pack_embedding(embedding), now, chunk_id) for chunk_id, embedding in rows],
            )

    def update_chunk_embeddings_draft(
        self,
        document_id: str,
        rows: Sequence[tuple[str, list[float]]],
        *,
        actor_id: str = "local-admin",
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.executemany(
                "UPDATE chunks SET embedding = ?, updated_at = ? WHERE id = ?",
                [(pack_embedding(embedding), now, chunk_id) for chunk_id, embedding in rows],
            )
            self._mark_document_draft_in_transaction(
                connection,
                document_id,
                actor_id,
                {"fields": ["embeddings"], "chunk_count": len(rows)},
            )

    def replace_media_assets(
        self,
        document_id: str,
        assets: list[dict[str, Any]],
        *,
        mark_draft: bool = False,
        actor_id: str = "local-admin",
    ) -> dict[str, int]:
        """Merge extraction results without overwriting operator-maintained fields."""

        now = utc_now()
        summary = {"new": 0, "changed": 0, "unchanged": 0, "missing": 0}
        with self.connect() as connection:
            existing_rows = connection.execute(
                "SELECT * FROM media_assets WHERE document_id = ?", (document_id,)
            ).fetchall()
            existing = {str(row["image_key"]): row for row in existing_rows}
            seen_keys: set[str] = set()
            for asset in assets:
                image_key = str(asset["image_key"])
                seen_keys.add(image_key)
                old = existing.get(image_key)
                extracted_caption = str(asset.get("caption") or "")
                if old is None:
                    summary["new"] += 1
                    connection.execute(
                        """
                        INSERT INTO media_assets (
                            id, document_id, image_key, filename, stored_path, mime_type,
                            sha256, width, height, source_kind, page, sheet, row_start,
                            row_end, cell_ref, caption, alt_text, product_ids, search_text,
                            enabled, review_status, extraction_status, extracted_caption,
                            created_at, updated_at
                        ) VALUES (
                            :id, :document_id, :image_key, :filename, :stored_path,
                            :mime_type, :sha256, :width, :height, :source_kind, :page,
                            :sheet, :row_start, :row_end, :cell_ref, :caption, :alt_text,
                            :product_ids, :search_text, :enabled, :review_status,
                            :extraction_status, :extracted_caption, :created_at, :updated_at
                        )
                        """,
                        {
                            **asset,
                            "document_id": document_id,
                            "caption": extracted_caption,
                            "alt_text": str(asset.get("alt_text") or extracted_caption),
                            "product_ids": json.dumps(
                                asset.get("product_ids", []), ensure_ascii=False
                            ),
                            "enabled": int(asset.get("enabled", True)),
                            "review_status": (
                                "pending" if mark_draft else asset.get("review_status", "approved")
                            ),
                            "extraction_status": "new" if mark_draft else "current",
                            "extracted_caption": extracted_caption,
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                    continue

                changed = (
                    any(
                        old[field] != asset.get(field)
                        for field in (
                            "filename",
                            "stored_path",
                            "mime_type",
                            "sha256",
                            "width",
                            "height",
                            "source_kind",
                            "page",
                            "sheet",
                            "row_start",
                            "row_end",
                            "cell_ref",
                            "search_text",
                        )
                    )
                    or str(old["extracted_caption"] or "") != extracted_caption
                )
                summary["changed" if changed else "unchanged"] += 1
                connection.execute(
                    """
                    UPDATE media_assets
                    SET filename = ?, stored_path = ?, mime_type = ?, sha256 = ?,
                        width = ?, height = ?, source_kind = ?, page = ?, sheet = ?,
                        row_start = ?, row_end = ?, cell_ref = ?, search_text = ?,
                        extracted_caption = ?, extraction_status = ?,
                        review_status = CASE WHEN ? THEN 'pending' ELSE review_status END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        asset["filename"],
                        asset["stored_path"],
                        asset["mime_type"],
                        asset["sha256"],
                        asset["width"],
                        asset["height"],
                        asset["source_kind"],
                        asset.get("page"),
                        asset.get("sheet"),
                        asset.get("row_start"),
                        asset.get("row_end"),
                        asset.get("cell_ref"),
                        asset.get("search_text", ""),
                        extracted_caption,
                        "changed" if changed and mark_draft else "current",
                        int(changed and mark_draft),
                        now,
                        old["id"],
                    ),
                )

            missing_keys = set(existing) - seen_keys
            summary["missing"] = len(missing_keys)
            if missing_keys:
                placeholders = ", ".join("?" for _ in missing_keys)
                if mark_draft:
                    connection.execute(
                        f"""
                        UPDATE media_assets
                        SET enabled = 0, review_status = 'pending',
                            extraction_status = 'missing', updated_at = ?
                        WHERE document_id = ? AND image_key IN ({placeholders})
                        """,  # noqa: S608
                        [now, document_id, *sorted(missing_keys)],
                    )
                else:
                    connection.execute(
                        f"""
                        DELETE FROM media_assets
                        WHERE document_id = ? AND image_key IN ({placeholders})
                        """,  # noqa: S608
                        [document_id, *sorted(missing_keys)],
                    )

            material_change = summary["new"] + summary["changed"] + summary["missing"] > 0
            if mark_draft and material_change:
                self._mark_document_draft_in_transaction(
                    connection,
                    document_id,
                    actor_id,
                    {"media_diff": summary},
                )
        return summary

    def list_media_assets(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, image_key, filename, mime_type, sha256, width,
                       height, source_kind, page, sheet, row_start, row_end, cell_ref,
                       caption, alt_text, product_ids, enabled, review_status,
                       extraction_status, extracted_caption, created_at, updated_at
                FROM media_assets
                WHERE document_id = ?
                ORDER BY COALESCE(page, 0), COALESCE(sheet, ''),
                         COALESCE(row_start, 0), image_key
                """,
                (document_id,),
            ).fetchall()
        return [_media_dict(row) for row in rows]

    def list_media_files(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, filename, stored_path, sha256, enabled
                FROM media_assets
                WHERE document_id = ?
                ORDER BY id
                """,
                (document_id,),
            ).fetchall()
        return [{**dict(row), "enabled": bool(row["enabled"])} for row in rows]

    def get_media_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM media_assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return _media_dict(row) if row else None

    def get_public_media_asset(self, asset_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT m.*
                FROM media_asset_versions m
                JOIN documents d
                  ON d.id = m.document_id AND d.active_version = m.version
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE m.asset_id = ?
                  AND m.enabled = 1
                  AND d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                """,
                (asset_id, PUBLIC_OWNER_COMPANY_ID, now, now),
            ).fetchone()
        if row is None:
            return None
        value = _media_dict(row)
        value["id"] = value.pop("asset_id")
        return value

    def fetch_public_media(self, document_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        now = utc_now()
        parameters: list[Any] = [PUBLIC_OWNER_COMPANY_ID, now, now]
        document_filter = ""
        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            document_filter = f" AND m.document_id IN ({placeholders})"
            parameters.extend(document_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT m.*, v.title AS document_title, d.filename AS document_filename
                FROM media_asset_versions m
                JOIN documents d
                  ON d.id = m.document_id AND d.active_version = m.version
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE m.enabled = 1
                  AND d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                  {document_filter}
                ORDER BY v.published_at DESC, COALESCE(m.page, 0),
                         COALESCE(m.row_start, 0), m.image_key
                """,  # noqa: S608
                parameters,
            ).fetchall()
        result = []
        for row in rows:
            value = _media_dict(row)
            value["id"] = value.pop("asset_id")
            value["source_version"] = value["version"]
            result.append(value)
        return result

    def are_media_snapshots_active(
        self,
        assets: Sequence[tuple[str, int, str]],
    ) -> bool:
        """Confirm that each returned media asset still belongs to the active snapshot."""

        if not assets:
            return False
        now = utc_now()
        with self.connect() as connection:
            for document_id, version, asset_id in assets:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM documents d
                    JOIN document_versions v
                      ON v.document_id = d.id AND v.version = d.active_version
                    JOIN media_asset_versions m
                      ON m.document_id = v.document_id AND m.version = v.version
                    WHERE d.id = ?
                      AND d.active_version = ?
                      AND m.asset_id = ?
                      AND m.enabled = 1
                      AND d.status = 'READY'
                      AND d.is_withdrawn = 0
                      AND v.owner_company_id = ?
                      AND v.visibility = 'public'
                      AND (v.effective_from IS NULL OR v.effective_from <= ?)
                      AND (v.effective_to IS NULL OR v.effective_to >= ?)
                    """,
                    (
                        document_id,
                        version,
                        asset_id,
                        PUBLIC_OWNER_COMPANY_ID,
                        now,
                        now,
                    ),
                ).fetchone()
                if row is None:
                    return False
        return True

    def update_media_asset(self, asset_id: str, values: dict[str, Any]) -> None:
        allowed = {"caption", "alt_text", "product_ids", "enabled", "review_status"}
        updates = {key: value for key, value in values.items() if key in allowed}
        if "product_ids" in updates:
            updates["product_ids"] = json.dumps(updates["product_ids"], ensure_ascii=False)
        if "enabled" in updates:
            updates["enabled"] = int(bool(updates["enabled"]))
        if not updates:
            return
        assignments = [f"{name} = ?" for name in updates]
        parameters = list(updates.values())
        assignments.append("updated_at = ?")
        parameters.extend([utc_now(), asset_id])
        with self.connect() as connection:
            connection.execute(
                f"UPDATE media_assets SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                parameters,
            )

    def update_media_asset_draft(
        self,
        asset_id: str,
        values: dict[str, Any],
        *,
        actor_id: str = "local-admin",
    ) -> None:
        allowed = {"caption", "alt_text", "product_ids", "enabled"}
        updates = {key: value for key, value in values.items() if key in allowed}
        if "product_ids" in updates:
            updates["product_ids"] = json.dumps(updates["product_ids"], ensure_ascii=False)
        if "enabled" in updates:
            updates["enabled"] = int(bool(updates["enabled"]))
        if not updates:
            return
        assignments = [f"{name} = ?" for name in updates]
        parameters = list(updates.values())
        assignments.extend(
            ["review_status = 'pending'", "extraction_status = 'edited'", "updated_at = ?"]
        )
        parameters.extend([utc_now(), asset_id])
        with self.connect() as connection:
            asset = connection.execute(
                "SELECT document_id FROM media_assets WHERE id = ?", (asset_id,)
            ).fetchone()
            if asset is None:
                raise KeyError(asset_id)
            connection.execute(
                f"UPDATE media_assets SET {', '.join(assignments)} WHERE id = ?",  # noqa: S608
                parameters,
            )
            self._mark_document_draft_in_transaction(
                connection,
                str(asset["document_id"]),
                actor_id,
                {"asset_id": asset_id, "fields": sorted(updates)},
            )

    def list_document_versions(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT v.*,
                       COUNT(DISTINCT c.chunk_id) AS chunk_count,
                       COUNT(DISTINCT m.asset_id) AS media_count,
                       CASE WHEN d.active_version = v.version THEN 1 ELSE 0 END AS is_active
                FROM document_versions v
                JOIN documents d ON d.id = v.document_id
                LEFT JOIN chunk_versions c
                  ON c.document_id = v.document_id AND c.version = v.version
                LEFT JOIN media_asset_versions m
                  ON m.document_id = v.document_id AND m.version = v.version
                WHERE v.document_id = ?
                GROUP BY v.document_id, v.version
                ORDER BY v.version DESC
                """,
                (document_id,),
            ).fetchall()
        result = []
        for row in rows:
            value = _document_dict(row)
            value["is_active"] = bool(row["is_active"])
            result.append(value)
        return result

    def list_publication_events(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, version, event_type, actor_id, details, created_at
                FROM publication_events
                WHERE document_id = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (document_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            try:
                value["details"] = json.loads(value["details"])
            except (TypeError, json.JSONDecodeError):
                value["details"] = {}
            result.append(value)
        return result

    def get_active_media_assets(self, document_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*
                FROM media_asset_versions m
                JOIN documents d
                  ON d.id = m.document_id AND d.active_version = m.version
                WHERE m.document_id = ?
                ORDER BY m.asset_id
                """,
                (document_id,),
            ).fetchall()
        result = []
        for row in rows:
            value = _media_dict(row)
            value["id"] = value.pop("asset_id")
            result.append(value)
        return result

    def update_active_media_paths(
        self,
        document_id: str,
        version: int,
        paths: dict[str, str],
    ) -> None:
        if not paths:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                UPDATE media_asset_versions
                SET stored_path = ?
                WHERE document_id = ? AND version = ? AND asset_id = ?
                """,
                [
                    (stored_path, document_id, version, asset_id)
                    for asset_id, stored_path in paths.items()
                ],
            )

    def are_source_snapshots_active(
        self,
        sources: Sequence[tuple[str, int, str]],
    ) -> bool:
        if not sources:
            return False
        now = utc_now()
        with self.connect() as connection:
            for document_id, version, chunk_id in sources:
                row = connection.execute(
                    """
                    SELECT 1
                    FROM documents d
                    JOIN document_versions v
                      ON v.document_id = d.id AND v.version = d.active_version
                    JOIN chunk_versions c
                      ON c.document_id = v.document_id AND c.version = v.version
                    WHERE d.id = ?
                      AND d.active_version = ?
                      AND c.chunk_id = ?
                      AND c.enabled = 1
                      AND d.is_withdrawn = 0
                      AND v.owner_company_id = ?
                      AND v.visibility = 'public'
                      AND (v.effective_from IS NULL OR v.effective_from <= ?)
                      AND (v.effective_to IS NULL OR v.effective_to >= ?)
                    """,
                    (
                        document_id,
                        version,
                        chunk_id,
                        PUBLIC_OWNER_COMPANY_ID,
                        now,
                        now,
                    ),
                ).fetchone()
                if row is None:
                    return False
        return True

    def delete_document(self, document_id: str) -> dict[str, Any] | None:
        document = self.get_document(document_id)
        if document is None:
            return None
        with self.connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return document

    def fetch_searchable_chunks(self, domains: Sequence[str] | None = None) -> list[dict[str, Any]]:
        now = utc_now()
        parameters: list[Any] = [PUBLIC_OWNER_COMPANY_ID, now, now]
        domain_filter = ""
        if domains:
            placeholders = ", ".join("?" for _ in domains)
            domain_filter = f" AND v.knowledge_domain IN ({placeholders})"
            parameters.extend(domains)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.chunk_id AS id, c.document_id, c.position, c.content,
                       c.page, c.sheet, c.row_start, c.row_end, c.embedding, c.enabled,
                       v.title AS document_title, d.filename,
                       v.knowledge_domain, v.version AS source_version
                FROM chunk_versions c
                JOIN documents d
                  ON d.id = c.document_id AND d.active_version = c.version
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE c.enabled = 1
                  AND d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                  {domain_filter}
                ORDER BY v.published_at DESC, c.position
                """,  # noqa: S608
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def log_query(self, values: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO query_logs (
                    id, question, status, source_ids, retrieval_ms, generation_ms,
                    reason_code, policy_version, evidence_status, fallback_kind,
                    source_snapshots, model_name, prompt_version, topic_check_ms,
                    initial_generation_ms, repair_ms, repair_attempts, repair_reason,
                    queue_ms, total_ms, cache_hit, created_at
                ) VALUES (
                    :id, :question, :status, :source_ids, :retrieval_ms,
                    :generation_ms, :reason_code, :policy_version, :evidence_status,
                    :fallback_kind, :source_snapshots, :model_name, :prompt_version,
                    :topic_check_ms, :initial_generation_ms, :repair_ms,
                    :repair_attempts, :repair_reason, :queue_ms, :total_ms,
                    :cache_hit, :created_at
                )
                """,
                values,
            )

    def update_query_runtime(self, request_id: str, *, queue_ms: int, total_ms: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE query_logs SET queue_ms = ?, total_ms = ? WHERE id = ?",
                (queue_ms, total_ms, request_id),
            )

    def public_knowledge_revision(self, domains: Sequence[str]) -> str:
        """Fingerprint active public versions, including time-window activation changes."""

        now = utc_now()
        parameters: list[Any] = [PUBLIC_OWNER_COMPANY_ID, now, now]
        domain_filter = ""
        if domains:
            placeholders = ", ".join("?" for _ in domains)
            domain_filter = f"AND v.knowledge_domain IN ({placeholders})"
            parameters.extend(domains)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT d.id, d.active_version, v.published_at
                FROM documents d
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                  {domain_filter}
                ORDER BY d.id
                """,  # noqa: S608
                parameters,
            ).fetchall()
        fingerprint = json.dumps([tuple(row) for row in rows], ensure_ascii=False)
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def stats(self) -> dict[str, int]:
        now = utc_now()
        with self.connect() as connection:
            documents = connection.execute(
                """
                SELECT COUNT(*)
                FROM documents d
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                """,
                (PUBLIC_OWNER_COMPANY_ID, now, now),
            ).fetchone()[0]
            chunks = connection.execute(
                """
                SELECT COUNT(*) FROM chunk_versions c
                JOIN documents d
                  ON d.id = c.document_id AND d.active_version = c.version
                JOIN document_versions v
                  ON v.document_id = d.id AND v.version = d.active_version
                WHERE c.enabled = 1
                  AND d.status = 'READY'
                  AND d.is_withdrawn = 0
                  AND v.owner_company_id = ?
                  AND v.visibility = 'public'
                  AND (v.effective_from IS NULL OR v.effective_from <= ?)
                  AND (v.effective_to IS NULL OR v.effective_to >= ?)
                """,
                (PUBLIC_OWNER_COMPANY_ID, now, now),
            ).fetchone()[0]
        return {"documents": int(documents), "chunks": int(chunks)}
