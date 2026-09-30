from pathlib import Path

from company_agent.database import KnowledgeRepository, pack_embedding, utc_now


def test_repository_document_and_chunk_lifecycle(tmp_path: Path) -> None:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    repository.initialize()
    now = utc_now()
    repository.create_document(
        {
            "id": "doc-1",
            "title": "产品手册",
            "filename": "manual.pdf",
            "stored_path": str(tmp_path / "manual.pdf"),
            "file_type": "pdf",
            "size_bytes": 100,
            "sha256": "abc123",
            "status": "PROCESSING",
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    repository.replace_chunks(
        "doc-1",
        [
            {
                "id": "chunk-1",
                "position": 0,
                "content": "测试知识",
                "page": 1,
                "sheet": None,
                "row_start": None,
                "row_end": None,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )
    repository.update_document_status("doc-1", "READY")

    assert repository.stats() == {"documents": 0, "chunks": 0}
    assert repository.fetch_searchable_chunks() == []

    repository.update_document_metadata(
        "doc-1",
        {
            "knowledge_domain": "product_public",
            "review_status": "approved",
            "publication_status": "published",
            "reviewer_id": "tester",
            "reviewed_at": now,
        },
    )
    repository.publish_document("doc-1", reviewer_id="tester")

    assert repository.stats() == {"documents": 1, "chunks": 1}
    assert len(repository.fetch_searchable_chunks(["product_public"])) == 1
    assert repository.fetch_searchable_chunks(["company_public"]) == []
    assert repository.get_document("doc-1")["chunk_count"] == 1  # type: ignore[index]
    assert repository.list_chunks("doc-1")[0]["enabled"] is True

    repository.update_chunk_draft("chunk-1", content="新知识", enabled=False)
    assert repository.get_chunk("chunk-1")["content"] == "新知识"  # type: ignore[index]
    assert repository.stats()["chunks"] == 1

    repository.publish_document("doc-1", reviewer_id="tester")
    assert repository.stats()["chunks"] == 0


def test_content_change_returns_document_to_review(tmp_path: Path) -> None:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    repository.initialize()
    now = utc_now()
    repository.create_document(
        {
            "id": "doc-1",
            "title": "产品手册",
            "filename": "manual.pdf",
            "stored_path": str(tmp_path / "manual.pdf"),
            "file_type": "pdf",
            "size_bytes": 100,
            "sha256": "change-test",
            "status": "READY",
            "publication_status": "published",
            "review_status": "approved",
            "created_at": now,
            "updated_at": now,
        }
    )

    repository.mark_document_content_changed("doc-1")
    document = repository.get_document("doc-1")

    assert document is not None
    assert document["publication_status"] == "draft"
    assert document["review_status"] == "pending"
    assert document["source_version"] == 2


def _seed_versioned_document(repository: KnowledgeRepository, tmp_path: Path) -> None:
    now = utc_now()
    repository.create_document(
        {
            "id": "versioned-doc",
            "title": "已发布标题",
            "filename": "facts.md",
            "stored_path": str(tmp_path / "facts.md"),
            "file_type": "md",
            "size_bytes": 100,
            "sha256": "versioned-document",
            "status": "READY",
            "created_at": now,
            "updated_at": now,
        }
    )
    repository.replace_chunks(
        "versioned-doc",
        [
            {
                "id": "versioned-chunk",
                "position": 0,
                "content": "已发布内容",
                "page": None,
                "sheet": None,
                "row_start": None,
                "row_end": None,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )
    repository.publish_document("versioned-doc", reviewer_id="tester")


def test_draft_edits_keep_active_snapshot_until_atomic_publish(tmp_path: Path) -> None:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    repository.initialize()
    _seed_versioned_document(repository, tmp_path)

    repository.update_document_draft("versioned-doc", {"title": "待审新标题"})
    repository.update_chunk_draft(
        "versioned-chunk",
        content="待审新内容",
        embedding=[0.0, 1.0],
    )

    draft = repository.get_document("versioned-doc")
    active = repository.fetch_searchable_chunks()
    assert draft is not None
    assert draft["source_version"] == 3
    assert draft["active_version"] == 1
    assert draft["review_status"] == "pending"
    assert active[0]["document_title"] == "已发布标题"
    assert active[0]["content"] == "已发布内容"

    repository.publish_document("versioned-doc", reviewer_id="tester")
    published = repository.fetch_searchable_chunks()
    assert published[0]["document_title"] == "待审新标题"
    assert published[0]["content"] == "待审新内容"
    assert published[0]["source_version"] == 3

    repository.rollback_document("versioned-doc", 1, actor_id="tester")
    rolled_back = repository.fetch_searchable_chunks()
    assert rolled_back[0]["document_title"] == "已发布标题"
    assert rolled_back[0]["content"] == "已发布内容"
    assert repository.list_document_versions("versioned-doc")[0]["version"] == 3
    assert repository.list_publication_events("versioned-doc")[0]["event_type"] == ("rolled_back")


def test_public_queries_filter_non_owned_published_snapshot(tmp_path: Path) -> None:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    repository.initialize()
    _seed_versioned_document(repository, tmp_path)

    repository.update_document_draft("versioned-doc", {"owner_company_id": "third-party-company"})
    assert len(repository.fetch_searchable_chunks()) == 1

    repository.publish_document("versioned-doc", reviewer_id="tester")
    assert repository.fetch_searchable_chunks() == []
    assert repository.stats() == {"documents": 0, "chunks": 0}


def test_media_edits_and_reextraction_preserve_published_and_operator_values(
    tmp_path: Path,
) -> None:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    repository.initialize()
    _seed_versioned_document(repository, tmp_path)
    base_asset = {
        "id": "asset-1",
        "image_key": "product-image",
        "filename": "product.png",
        "stored_path": str(tmp_path / "product.png"),
        "mime_type": "image/png",
        "sha256": "sha-v1",
        "width": 800,
        "height": 800,
        "source_kind": "product_image",
        "page": 1,
        "sheet": None,
        "row_start": None,
        "row_end": None,
        "cell_ref": None,
        "caption": "原始标题",
        "alt_text": "原始替代文本",
        "product_ids": ["P01"],
        "search_text": "P01 产品图",
    }
    repository.replace_media_assets("versioned-doc", [base_asset], mark_draft=True)
    repository.publish_document("versioned-doc", reviewer_id="tester")

    repository.update_media_asset_draft("asset-1", {"caption": "运营修订标题"})
    assert repository.fetch_public_media()[0]["caption"] == "原始标题"

    changed_asset = {
        **base_asset,
        "id": "new-extraction-id",
        "sha256": "sha-v2",
        "caption": "重新提取标题",
        "search_text": "P01 新原文产品图",
    }
    diff = repository.replace_media_assets("versioned-doc", [changed_asset], mark_draft=True)
    current = repository.list_media_assets("versioned-doc")[0]
    assert diff["changed"] == 1
    assert current["id"] == "asset-1"
    assert current["caption"] == "运营修订标题"
    assert current["extracted_caption"] == "重新提取标题"
    assert current["review_status"] == "pending"
    assert current["extraction_status"] == "changed"
    assert repository.fetch_public_media()[0]["caption"] == "原始标题"


def test_initialize_migrates_legacy_published_row_to_snapshot(tmp_path: Path) -> None:
    database_path = tmp_path / "knowledge.db"
    repository = KnowledgeRepository(database_path)
    repository.initialize()
    now = utc_now()
    repository.create_document(
        {
            "id": "legacy-doc",
            "title": "旧公开资料",
            "filename": "legacy.md",
            "stored_path": str(tmp_path / "legacy.md"),
            "file_type": "md",
            "size_bytes": 10,
            "sha256": "legacy-published",
            "status": "READY",
            "publication_status": "published",
            "review_status": "approved",
            "created_at": now,
            "updated_at": now,
        }
    )
    repository.replace_chunks(
        "legacy-doc",
        [
            {
                "id": "legacy-chunk",
                "position": 0,
                "content": "旧版本公开内容",
                "page": None,
                "sheet": None,
                "row_start": None,
                "row_end": None,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )

    repository.initialize()

    migrated = repository.get_document("legacy-doc")
    assert migrated is not None
    assert migrated["active_version"] == 1
    assert repository.fetch_searchable_chunks()[0]["content"] == "旧版本公开内容"
