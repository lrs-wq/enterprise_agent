import asyncio
from pathlib import Path
from typing import Literal

import pytest

from company_agent.config import Settings
from company_agent.database import KnowledgeRepository, pack_embedding, utc_now
from company_agent.ollama_client import OllamaTimeoutError, OllamaUnavailableError
from company_agent.retrieval import SearchResult, retrieve
from company_agent.schemas import (
    AnswerSection,
    ChatMessage,
    ChatRequest,
    GeneralKnowledgeAnswer,
    GeneratedAnswer,
    GeneratedSection,
    QuestionTopicClassification,
    SourceRef,
)
from company_agent.services import AgentService, KnowledgeService, ModelStageBudgetExceeded


class FakeOllama:
    def __init__(
        self,
        answer: GeneratedAnswer | None = None,
        general_answer: GeneralKnowledgeAnswer | None = None,
        topic_kind: Literal[
            "company",
            "pet_product",
            "pet_care",
            "medical",
            "third_party",
            "pet_other",
            "non_pet",
            "uncertain",
        ] = "pet_care",
        use_prior_pet_subject: bool = False,
        referenced_pet_subject: str | None = None,
    ) -> None:
        self.generated = answer
        self.general_generated = general_answer
        self.topic_kind = topic_kind
        self.use_prior_pet_subject = use_prior_pet_subject
        self.referenced_pet_subject = referenced_pet_subject
        self.embed_calls = 0
        self.answer_calls = 0
        self.repair_calls = 0
        self.general_calls = 0
        self.general_questions: list[str] = []
        self.general_repair_calls = 0
        self.topic_calls = 0
        self.topic_questions: list[str] = []
        self.topic_prior_subjects: list[str | None] = []
        self.topic_recent_user_questions: list[tuple[str, ...]] = []
        self.answer_questions: list[str] = []
        self.repair_questions: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    async def answer(self, question: str, *_args: object, **_kwargs: object) -> GeneratedAnswer:
        self.answer_calls += 1
        self.answer_questions.append(question)
        assert self.generated is not None
        return self.generated

    async def repair_answer(
        self, question: str, *_args: object, **_kwargs: object
    ) -> GeneratedAnswer:
        self.repair_calls += 1
        self.repair_questions.append(question)
        assert self.generated is not None
        return self.generated

    async def answer_general_knowledge(
        self, question: str, *_args: object, **_kwargs: object
    ) -> GeneralKnowledgeAnswer:
        self.general_calls += 1
        self.general_questions.append(question)
        assert self.general_generated is not None
        return self.general_generated

    async def repair_general_knowledge(
        self, *_args: object, **_kwargs: object
    ) -> GeneralKnowledgeAnswer:
        self.general_repair_calls += 1
        assert self.general_generated is not None
        return self.general_generated

    async def classify_question_topic(
        self,
        _question: str,
        _company_name: str,
        _owned_aliases: tuple[str, ...],
        prior_pet_subject: str | None = None,
        recent_user_questions: tuple[str, ...] = (),
    ) -> QuestionTopicClassification:
        self.topic_calls += 1
        self.topic_questions.append(_question)
        self.topic_prior_subjects.append(prior_pet_subject)
        self.topic_recent_user_questions.append(recent_user_questions)
        return QuestionTopicClassification(
            kind=self.topic_kind,
            use_prior_pet_subject=self.use_prior_pet_subject,
            referenced_pet_subject=self.referenced_pet_subject,
        )


def seed_published_product(repository: KnowledgeRepository, tmp_path: Path) -> None:
    now = utc_now()
    repository.create_document(
        {
            "id": "doc-1",
            "title": "CKIN PET 产品资料",
            "filename": "products.xlsx",
            "stored_path": str(tmp_path / "products.xlsx"),
            "file_type": "xlsx",
            "size_bytes": 100,
            "sha256": "service-test",
            "status": "READY",
            "knowledge_domain": "product_public",
            "publication_status": "published",
            "review_status": "approved",
            "reviewed_at": now,
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
                "content": "CKIN PET 智能喂食器支持定时定量投喂。",
                "page": None,
                "sheet": "产品表",
                "row_start": 2,
                "row_end": 2,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )
    repository.publish_document("doc-1", reviewer_id="tester")


def seed_published_capacity_fact(
    repository: KnowledgeRepository,
    tmp_path: Path,
    *,
    suffix: str,
    content: str,
) -> None:
    now = utc_now()
    document_id = f"capacity-{suffix}"
    repository.create_document(
        {
            "id": document_id,
            "title": f"F02 容量资料 {suffix}",
            "filename": f"capacity-{suffix}.txt",
            "stored_path": str(tmp_path / f"capacity-{suffix}.txt"),
            "file_type": "txt",
            "size_bytes": len(content.encode("utf-8")),
            "sha256": f"capacity-{suffix}",
            "status": "READY",
            "knowledge_domain": "product_public",
            "publication_status": "draft",
            "review_status": "pending",
            "created_at": now,
            "updated_at": now,
        }
    )
    repository.replace_chunks(
        document_id,
        [
            {
                "id": f"capacity-chunk-{suffix}",
                "position": 0,
                "content": content,
                "page": 1,
                "sheet": None,
                "row_start": None,
                "row_end": None,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )
    repository.publish_document(document_id, reviewer_id="tester")


def grounded_answer() -> GeneratedAnswer:
    return GeneratedAnswer(
        status="grounded",
        intent="product_intro",
        title="智能喂食器",
        summary="CKIN PET 智能喂食器支持定时定量投喂。",
        summary_source_ids=["S1"],
        sections=[
            GeneratedSection(
                heading="核心功能",
                content="智能喂食器支持定时定量投喂。",
                source_ids=["S1"],
            )
        ],
        follow_up_questions=[],
    )


def general_answer() -> GeneralKnowledgeAnswer:
    return GeneralKnowledgeAnswer(
        title="猫砂盆日常清洁",
        summary="日常及时清理结团和排泄物，并保持猫砂盆干燥、通风。",
        sections=[
            AnswerSection(
                heading="清洁原则",
                content="根据使用情况及时清理，定期更换猫砂并清洗、彻底晾干猫砂盆。",
            )
        ],
        follow_up_questions=[],
    )


async def test_grounded_response_uses_only_visible_validated_sources(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))

    assert response.status == "grounded"
    assert response.evidence_status == "supported"
    assert [source.source_id for source in response.sources] == ["S1"]
    assert response.answer.summary.endswith("[S1]")
    assert response.answer.sections[0].content.endswith("[S1]")
    assert response.sources[0].excerpt == "CKIN PET 智能喂食器支持定时定量投喂。"
    assert response.confidence is None
    assert ollama.repair_calls == 0


async def test_verified_public_answer_cache_reuses_only_context_free_active_sources(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(question="CKIN PET 喂食器有什么功能？")

    first = await service.chat(request)
    second = await service.chat(request)

    assert first.status == second.status == "grounded"
    assert first.request_id != second.request_id
    assert ollama.answer_calls == 1
    assert second.metrics["cache_hit"] == 1
    assert second.metrics["generation_ms"] == 0
    assert second.sources == first.sources
    repository.update_query_runtime(second.request_id, queue_ms=7, total_ms=18)
    with repository.connect() as connection:
        row = connection.execute(
            """
            SELECT source_snapshots, model_name, prompt_version, cache_hit,
                   queue_ms, total_ms
            FROM query_logs WHERE id = ?
            """,
            (second.request_id,),
        ).fetchone()
    assert row is not None
    assert '"document_id": "doc-1"' in row["source_snapshots"]
    assert row["model_name"] == settings.chat_model
    assert row["prompt_version"]
    assert (row["cache_hit"], row["queue_ms"], row["total_ms"]) == (1, 7, 18)

    repository.update_document_draft("doc-1", {"title": "待审核的新标题"})
    while_draft = await service.chat(request)
    assert while_draft.metrics["cache_hit"] == 1
    assert ollama.answer_calls == 1
    repository.publish_document("doc-1", reviewer_id="test-reviewer")
    after_publish = await service.chat(request)
    assert after_publish.status == "grounded"
    assert after_publish.metrics.get("cache_hit", 0) == 0
    assert ollama.answer_calls == 2

    repository.withdraw_document("doc-1")
    after_withdrawal = await service.chat(request)
    assert after_withdrawal.status == "insufficient_evidence"
    assert after_withdrawal.metrics.get("cache_hit", 0) == 0


async def test_public_answer_cache_never_reuses_user_history(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="CKIN PET 喂食器有什么功能？",
        history=[ChatMessage(role="user", content="我想给自己的宠物选设备")],
    )

    await service.chat(request)
    second = await service.chat(request)

    assert ollama.answer_calls == 2
    assert second.metrics.get("cache_hit", 0) == 0


async def test_generic_pet_supplies_question_retrieves_company_products(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer(), topic_kind="pet_product")
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="宠物用品有什么"))

    assert response.status == "grounded"
    assert response.intent == "product_intro"
    assert response.decision.reason_code == "PET_PRODUCT_CATEGORY_ALLOWED"
    assert response.answer.summary.endswith("[S1]")
    assert [source.source_id for source in response.sources] == ["S1"]
    assert ollama.topic_calls == 1
    assert ollama.embed_calls == 1
    assert ollama.answer_calls == 1
    assert ollama.general_calls == 0


@pytest.mark.parametrize(
    ("question", "topic_kind", "route_reason"),
    [
        ("CKIN PET 喂食器有什么功能？", "pet_product", "OWNED_PRODUCT_ALLOWED"),
        ("猫砂盆多久清洁一次？", "pet_care", "PET_CARE_GENERAL_MODEL_KNOWLEDGE"),
    ],
)
async def test_model_unavailable_preserves_original_route(
    tmp_path: Path,
    question: str,
    topic_kind: Literal["pet_product", "pet_care"],
    route_reason: str,
) -> None:
    class UnavailableOllama(FakeOllama):
        async def answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            raise OllamaUnavailableError("生成服务不可用")

        async def answer_general_knowledge(
            self, *_args: object, **_kwargs: object
        ) -> GeneralKnowledgeAnswer:
            raise OllamaUnavailableError("通用知识服务不可用")

    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = AgentService(settings, repository, UnavailableOllama(topic_kind=topic_kind))  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question=question))

    assert response.status == "service_unavailable"
    assert response.decision.reason_code == "LOCAL_MODEL_UNAVAILABLE"
    assert response.decision.route_reason_code == route_reason
    assert response.fallback_kind == "service_error"


async def test_model_can_route_company_wording_to_product_evidence(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer(), topic_kind="pet_product")
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="公司的喂食设备有什么"))

    assert response.status == "grounded"
    assert response.intent == "product_intro"
    assert response.decision.route_reason_code == "PET_PRODUCT_CATEGORY_ALLOWED"
    assert [source.source_id for source in response.sources] == ["S1"]
    assert ollama.topic_calls == 1
    assert ollama.embed_calls == 1
    assert ollama.general_calls == 0


@pytest.mark.parametrize(
    ("question", "kind", "reason_code"),
    [
        ("狗不舒服怎么办？", "medical", "PET_MEDICAL_ADVICE"),
        ("华为的宠物用品怎么样？", "third_party", "THIRD_PARTY_BRAND"),
        ("宠物用品有什么", "non_pet", "UNRELATED_TOPIC"),
    ],
)
async def test_model_topic_can_block_an_ordinary_question(
    tmp_path: Path,
    question: str,
    kind: Literal["medical", "third_party", "non_pet"],
    reason_code: str,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(topic_kind=kind)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question=question))

    assert response.status == "out_of_scope"
    assert response.decision.reason_code == reason_code
    assert response.decision.route_reason_code == reason_code
    assert ollama.topic_calls == 1
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0
    assert ollama.general_calls == 0


async def test_unlisted_pet_product_wording_retrieves_company_products(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer(), topic_kind="pet_product")
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="毛孩子有哪些喂食设备"))

    assert response.status == "grounded"
    assert response.intent == "product_intro"
    assert response.decision.reason_code == "PET_PRODUCT_CATEGORY_ALLOWED"
    assert [source.source_id for source in response.sources] == ["S1"]
    assert response.metrics["topic_check_ms"] >= 0
    assert ollama.topic_calls == 1
    assert ollama.embed_calls == 1
    assert ollama.answer_calls == 1
    assert ollama.general_calls == 0


async def test_model_timeout_propagates_to_http_runtime_layer(tmp_path: Path) -> None:
    class TimeoutOllama(FakeOllama):
        async def embed(self, _texts: list[str]) -> list[list[float]]:
            raise OllamaTimeoutError("embedding timed out")

    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = AgentService(settings, repository, TimeoutOllama())  # type: ignore[arg-type]

    with pytest.raises(OllamaTimeoutError):
        await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))


async def test_mixed_brand_comparison_answers_only_owned_product(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="小佩和 CKIN PET 的喂食器哪个好？"))

    assert response.status == "partial_answer"
    assert response.evidence_status == "partial"
    assert response.fallback_kind is None
    assert response.decision.reason_code == "SELF_PRODUCT_ONLY_PARTIAL"
    assert response.answer.title == "智能喂食器"
    assert "小佩并非本公司或旗下自有品牌产品" in response.answer.summary
    assert "不对小佩作评价或比较" in response.answer.summary
    assert response.answer.refusal_reason is not None
    assert "未经过本公司知识库的审核与发布" in response.answer.refusal_reason
    assert response.answer.sections[0].heading == "自有产品信息"
    assert response.answer.sections[0].content.endswith("[S1]")
    assert [source.source_id for source in response.sources] == ["S1"]
    assert "小佩" not in ollama.answer_questions[0]
    assert "CKIN PET" in ollama.answer_questions[0]
    assert "喂食器" in ollama.answer_questions[0]
    assert ollama.repair_calls == 0


def test_scope_only_fallback_is_not_reported_as_effective_partial_answer(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    decision = service.boundary.classify("小佩和 CKIN PET 的喂食器哪个好？")

    response = service._self_only_safe_fallback(  # noqa: SLF001
        "request-id",
        decision,
        reason_code="SELF_ONLY_SAFE_FALLBACK_TEST",
        retrieval_ms=1,
        generation_ms=2,
        retrieved_chunks=3,
    )

    assert response.status == "partial_answer"
    assert response.evidence_status == "not_applicable"
    assert response.fallback_kind == "scope_only"
    assert "没有形成可验证的实质公司或产品答案" in response.answer.sections[0].content


def test_medical_partial_fallback_keeps_medical_copy(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    decision = service.boundary.classify("请介绍 CKIN PET F02 饮水机，另外猫咪呕吐了怎么办？")

    scope_only = service._self_only_safe_fallback(  # noqa: SLF001
        "request-id",
        decision,
        reason_code="SELF_ONLY_SAFE_FALLBACK_TEST",
        retrieval_ms=1,
        generation_ms=2,
        retrieved_chunks=3,
    )
    missing = service._insufficient_response(  # noqa: SLF001
        "request-id",
        decision,
        "没有找到直接支持该产品问题的资料。",
        reason_code="NO_DIRECT_EVIDENCE",
    )

    assert decision.partial_kind == "medical"
    assert scope_only.answer.title == "可回答独立的产品问题"
    assert "兽医" in scope_only.answer.summary
    assert "第三方品牌" not in " ".join(step.summary for step in scope_only.thinking_summary)
    assert missing.answer.title == "可回答独立的产品问题"
    assert "兽医" in missing.answer.summary


async def test_mixed_brand_guard_failure_uses_self_evidence(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    unsafe = grounded_answer().model_copy(update={"summary": "小佩与 CKIN PET 各有特点。"})
    ollama = FakeOllama(unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="小佩和 CKIN PET 的喂食器哪个好？"))

    assert response.status == "partial_answer"
    assert response.evidence_status == "partial"
    assert response.answer.title == "已审核资料中的相关信息"
    assert response.answer.refusal_reason is not None
    assert response.sources
    assert response.decision.reason_code == "SELF_PRODUCT_ONLY_PARTIAL"
    assert response.metrics["extractive_fallback"] == 1
    assert response.fallback_kind == "verified_extractive"
    evidence_text = "\n".join(section.content for section in response.answer.sections)
    assert "小佩" not in evidence_text
    assert "不对小佩作评价或比较" in response.answer.summary
    assert ollama.repair_calls == 1
    assert "小佩" not in ollama.repair_questions[0]


async def test_mixed_brand_invalid_model_json_uses_self_evidence(tmp_path: Path) -> None:
    class InvalidJsonOllama(FakeOllama):
        async def answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            raise RuntimeError("模型没有返回合法的结构化结果")

    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = AgentService(
        settings,
        repository,
        InvalidJsonOllama(),  # type: ignore[arg-type]
    )

    response = await service.chat(
        ChatRequest(question="CKIN PET vs Homerunpet，哪个 feeder 更好？")
    )

    assert response.status == "partial_answer"
    assert response.evidence_status == "partial"
    assert response.sources
    assert response.metrics["extractive_fallback"] == 1
    assert response.fallback_kind == "verified_extractive"
    assert "Homerunpet" not in "".join(section.content for section in response.answer.sections)


async def test_mixed_brand_generic_third_party_leak_uses_self_evidence(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    leaked = grounded_answer().model_copy(
        update={"summary": "第三方品牌产品与 CKIN PET 各有特点。"}
    )
    service = AgentService(
        settings,
        repository,
        FakeOllama(leaked),  # type: ignore[arg-type]
    )

    response = await service.chat(ChatRequest(question="CKIN PET 和小佩哪个好？"))

    assert response.status == "partial_answer"
    assert response.evidence_status == "partial"
    assert response.sources
    assert response.metrics["extractive_fallback"] == 1
    assert "第三方品牌" not in "".join(section.content for section in response.answer.sections)


async def test_product_image_request_returns_verified_media_without_generation(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    repository.replace_media_assets(
        "doc-1",
        [
            {
                "id": "media-1",
                "image_key": "product-feeder",
                "filename": "feeder.png",
                "stored_path": str(settings.media_dir / "doc-1" / "feeder.png"),
                "mime_type": "image/png",
                "sha256": "image-sha",
                "width": 800,
                "height": 800,
                "source_kind": "product_image",
                "page": None,
                "sheet": "产品表",
                "row_start": 2,
                "row_end": 2,
                "cell_ref": "A2",
                "caption": "CKIN PET 智能喂食器",
                "alt_text": "CKIN PET 智能喂食器产品图",
                "product_ids": ["X01"],
                "search_text": "CKIN PET X01 智能喂食器 定时定量投喂",
            },
            {
                "id": "media-unsafe",
                "image_key": "unsafe-third-party-caption",
                "filename": "unsafe.png",
                "stored_path": str(settings.media_dir / "doc-1" / "unsafe.png"),
                "mime_type": "image/png",
                "sha256": "unsafe-image-sha",
                "width": 1200,
                "height": 1200,
                "source_kind": "product_image",
                "page": None,
                "sheet": "产品表",
                "row_start": 2,
                "row_end": 2,
                "cell_ref": "B2",
                "caption": "小佩智能喂食器",
                "alt_text": "小佩 X01 智能喂食器产品图",
                "product_ids": ["X01"],
                "search_text": "CKIN PET X01 智能喂食器 定时定量投喂",
            },
        ],
        mark_draft=True,
    )
    repository.publish_document("doc-1", reviewer_id="tester")
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET X01 喂食器有图片吗？"))

    assert response.status == "grounded"
    assert response.decision.reason_code == "PRODUCT_MEDIA_FOUND"
    assert response.images[0].id == "media-1"
    assert response.images[0].url == "/api/media/media-1"
    assert response.images[0].source_id == "M1"
    assert response.images[0].product_keys == ["ckin-pet:pet-feeder:X01:standard"]
    assert [image.id for image in response.images] == ["media-1"]
    assert response.sources[0].source_kind == "media"
    assert response.sources[0].chunk_id == "media:media-1"
    assert response.sources[0].row_start == 2
    assert response.metrics["generation_ms"] == 0
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0


async def test_media_catalog_separates_same_model_by_category_and_variant(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    assets = [
        {
            "id": "media-f02-water",
            "image_key": "f02-water",
            "filename": "water.png",
            "stored_path": str(settings.media_dir / "doc-1" / "water.png"),
            "mime_type": "image/png",
            "sha256": "water-sha",
            "width": 800,
            "height": 800,
            "source_kind": "product_image",
            "page": None,
            "sheet": "产品表",
            "row_start": 9,
            "row_end": 9,
            "cell_ref": "A9",
            "caption": "F02饮水机-WIFI款",
            "alt_text": "F02 Water Dispenser - WiFi Version",
            "product_ids": ["F02"],
            "search_text": "CKIN PET F02饮水机-WIFI款",
        },
        {
            "id": "media-f02-feeder",
            "image_key": "f02-feeder",
            "filename": "feeder.png",
            "stored_path": str(settings.media_dir / "doc-1" / "feeder.png"),
            "mime_type": "image/png",
            "sha256": "feeder-sha",
            "width": 800,
            "height": 800,
            "source_kind": "product_image",
            "page": None,
            "sheet": "产品表",
            "row_start": 13,
            "row_end": 13,
            "cell_ref": "A13",
            "caption": "F02喂食器-视频款",
            "alt_text": "F02 Pet Feeder - Video Version",
            "product_ids": ["F02"],
            "search_text": "CKIN PET F02喂食器-视频款",
        },
    ]
    repository.replace_media_assets("doc-1", assets, mark_draft=True)
    repository.publish_document("doc-1", reviewer_id="tester")
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="请展示 CKIN PET F02 视频款喂食器图片"))

    assert [image.id for image in response.images] == ["media-f02-feeder"]
    assert response.images[0].row_start == 13
    assert response.sources[0].source_kind == "media"
    assert response.sources[0].row_start == 13
    assert "water-dispenser" not in " ".join(response.images[0].product_keys)
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0

    water_evidence = SearchResult(
        source=SourceRef(
            source_id="S1",
            chunk_id="water-chunk",
            document_id="doc-1",
            document_title="CKIN PET 产品资料",
            filename="products.xlsx",
            sheet="产品表",
            row_start=9,
            row_end=9,
            score=1.0,
            excerpt="CKIN PET F02 饮水机已确认功能。",
            source_version=2,
        ),
        content="CKIN PET F02 饮水机已确认功能。",
    )
    combined_images = service._select_response_images(  # noqa: SLF001
        ChatRequest(question="请用图片配合说明 CKIN PET F02 的外观和已确认功能"),
        [water_evidence],
    )

    assert [image.id for image in combined_images] == ["media-f02-water"]


def test_image_intent_does_not_treat_generic_look_as_image_request(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )

    assert service._response_content_mode("帮我看一下 F02 的参数") == "text_only"  # noqa: SLF001
    assert service._response_content_mode("看一下 F02 的产品图片") == "image_only"  # noqa: SLF001
    assert (
        service._response_content_mode("介绍 F02 的参数并展示图片")  # noqa: SLF001
        == "text_and_image"
    )


async def test_missing_structured_source_ids_are_not_inferred_and_answer_is_downgraded(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    invalid = grounded_answer().model_copy(
        update={
            "summary": "CKIN PET 智能喂食器支持定时定量投喂。",
            "summary_source_ids": [],
            "sections": [
                GeneratedSection(
                    heading="功能",
                    content="支持定时定量投喂。",
                    source_ids=[],
                )
            ],
        }
    )
    ollama = FakeOllama(invalid)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))

    assert response.status == "insufficient_evidence"
    assert response.sources == []
    assert "稳定绑定" in response.answer.summary
    assert response.decision.reason_code.startswith("EVIDENCE_GUARD_")
    assert response.follow_up_questions == ["CKIN PET 喂食器有什么功能？"]
    assert "补充" not in "".join(response.follow_up_questions)
    assert ollama.repair_calls == 1


async def test_contradicted_claim_is_repaired_once_then_downgraded(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    contradicted = grounded_answer().model_copy(
        update={
            "summary": "CKIN PET 智能喂食器不支持定时定量投喂。",
            "sections": [
                GeneratedSection(
                    heading="核心功能",
                    content="智能喂食器不支持定时定量投喂。",
                    source_ids=["S1"],
                )
            ],
        }
    )
    ollama = FakeOllama(contradicted)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器支持定时吗？"))

    assert response.status == "insufficient_evidence"
    assert response.evidence_status == "missing"
    assert response.sources == []
    assert response.decision.reason_code == ("EVIDENCE_GUARD_CONTRADICTED_CLAIM_POLARITY")
    assert ollama.repair_calls == 1


async def test_failed_repair_keeps_other_verified_claims(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    partially_invalid = grounded_answer().model_copy(
        update={
            "summary": ("CKIN PET 智能喂食器支持定时定量投喂。该产品拥有知识库未记载的24L容量。")
        }
    )
    ollama = FakeOllama(partially_invalid)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))

    assert response.status == "grounded"
    assert response.evidence_status == "supported"
    assert "定时定量投喂" in response.answer.summary
    assert "24L" not in response.answer.summary
    assert response.metrics["salvaged_claims"] == 1
    assert ollama.repair_calls == 1


async def test_failed_generation_uses_verified_extractive_fallback(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    unsafe = grounded_answer().model_copy(
        update={
            "summary": "CKIN PET 智能喂食器可以用于治疗宠物疾病。",
            "sections": [
                GeneratedSection(
                    heading="错误生成",
                    content="该产品能够治疗疾病。",
                    source_ids=["S1"],
                )
            ],
        }
    )
    ollama = FakeOllama(unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))

    assert response.status == "grounded"
    assert response.evidence_status == "supported"
    assert "定时定量投喂" in response.answer.summary
    assert "治疗" not in response.answer.summary
    assert [source.source_id for source in response.sources] == ["S1"]
    assert response.metrics["extractive_fallback"] == 1
    assert ollama.repair_calls == 1


async def test_slow_repair_returns_validated_initial_salvage_before_chat_deadline(
    tmp_path: Path,
) -> None:
    class SlowRepairOllama(FakeOllama):
        def __init__(self, answer: GeneratedAnswer) -> None:
            super().__init__(answer)
            self.repair_cancelled = asyncio.Event()

        async def repair_answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            self.repair_calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.repair_cancelled.set()
            raise AssertionError("repair should be cancelled")

    settings = Settings(
        data_dir=tmp_path / "data",
        min_retrieval_score=0.0,
        chat_finalize_reserve_seconds=1,
    )
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    invalid = grounded_answer().model_copy(
        update={"summary": "CKIN PET 智能喂食器支持定时定量投喂。该产品拥有未记载的24L容量。"}
    )
    ollama = SlowRepairOllama(invalid)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    deadline = asyncio.get_running_loop().time() + 1.5

    response = await service.chat(
        ChatRequest(question="CKIN PET 喂食器有什么功能？"), deadline=deadline
    )

    assert response.status == "grounded"
    assert response.metrics["repair_timed_out"] == 1
    assert response.metrics["salvaged_claims"] == 1
    assert "24L" not in response.answer.summary
    assert "定时定量投喂" in response.answer.summary
    assert ollama.repair_calls == 1
    assert ollama.repair_cancelled.is_set()
    assert asyncio.get_running_loop().time() < deadline


async def test_slow_repair_uses_verified_extractive_fallback(tmp_path: Path) -> None:
    class SlowRepairOllama(FakeOllama):
        async def repair_answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            self.repair_calls += 1
            await asyncio.Event().wait()
            raise AssertionError("repair should be cancelled")

    settings = Settings(
        data_dir=tmp_path / "data",
        min_retrieval_score=0.0,
        chat_finalize_reserve_seconds=1,
    )
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    unsafe = grounded_answer().model_copy(
        update={
            "summary": "CKIN PET 智能喂食器可以用于治疗宠物疾病。",
            "sections": [
                GeneratedSection(
                    heading="错误生成",
                    content="该产品能够治疗疾病。",
                    source_ids=["S1"],
                )
            ],
        }
    )
    ollama = SlowRepairOllama(unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    deadline = asyncio.get_running_loop().time() + 1.5

    response = await service.chat(
        ChatRequest(question="CKIN PET 喂食器有什么功能？"), deadline=deadline
    )

    assert response.status == "grounded"
    assert response.fallback_kind == "verified_extractive"
    assert response.metrics["repair_timed_out"] == 1
    assert response.metrics["extractive_fallback"] == 1
    assert "治疗" not in response.answer.summary
    assert response.sources[0].source_id == "S1"
    assert asyncio.get_running_loop().time() < deadline


async def test_slow_repair_without_safe_fallback_raises_stage_timeout(tmp_path: Path) -> None:
    class SlowRepairOllama(FakeOllama):
        async def repair_answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            self.repair_calls += 1
            await asyncio.Event().wait()
            raise AssertionError("repair should be cancelled")

    settings = Settings(
        data_dir=tmp_path / "data",
        min_retrieval_score=0.0,
        chat_finalize_reserve_seconds=1,
    )
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    invalid = grounded_answer().model_copy(update={"summary_source_ids": []})
    ollama = SlowRepairOllama(invalid)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    deadline = asyncio.get_running_loop().time() + 1.5

    with pytest.raises(ModelStageBudgetExceeded):
        await service.chat(
            ChatRequest(question="CKIN PET 喂食器到底有什么功能？"), deadline=deadline
        )

    assert ollama.repair_calls == 1


def test_timeout_salvage_must_keep_requested_product_and_attribute() -> None:
    answer = grounded_answer()

    assert AgentService._salvage_covers_requested_facts(  # noqa: SLF001
        "CKIN PET 喂食器有什么功能？", answer
    )
    assert not AgentService._salvage_covers_requested_facts(  # noqa: SLF001
        "CKIN PET F02 饮水机容量是多少？", answer
    )
    assert not AgentService._salvage_covers_requested_facts(  # noqa: SLF001
        "CKIN PET 喂食器售后流程是什么？", answer
    )


async def test_slow_initial_generation_uses_verified_extractive_fallback(
    tmp_path: Path,
) -> None:
    class SlowInitialOllama(FakeOllama):
        async def answer(self, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            self.answer_calls += 1
            await asyncio.Event().wait()
            raise AssertionError("answer should be cancelled")

    settings = Settings(
        data_dir=tmp_path / "data",
        min_retrieval_score=0.0,
        chat_finalize_reserve_seconds=1,
    )
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    ollama = SlowInitialOllama()
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    deadline = asyncio.get_running_loop().time() + 1.5

    response = await service.chat(
        ChatRequest(question="CKIN PET 喂食器有什么功能？"), deadline=deadline
    )

    assert response.status == "grounded"
    assert response.fallback_kind == "verified_extractive"
    assert response.metrics["answer_timed_out"] == 1
    assert response.metrics["repair_timed_out"] == 0
    assert ollama.answer_calls == 1
    assert ollama.repair_calls == 0
    assert asyncio.get_running_loop().time() < deadline


async def test_conflicting_capacity_sources_are_not_answered(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_capacity_fact(
        repository,
        tmp_path,
        suffix="4l",
        content="CKIN PET F02 饮水机容量为 4 升。",
    )
    seed_published_capacity_fact(
        repository,
        tmp_path,
        suffix="6l",
        content="CKIN PET F02 饮水机容量为 6 升。",
    )
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="F02 的容量到底是多少？"))

    assert response.status == "insufficient_evidence"
    assert response.evidence_status == "missing"
    assert response.decision.reason_code == "CONFLICTING_PUBLISHED_EVIDENCE"
    assert response.sources == []
    assert ollama.answer_calls == 0


async def test_extractive_fallback_requires_direct_standby_power_evidence(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_capacity_fact(
        repository,
        tmp_path,
        suffix="power",
        content="CKIN PET F02 饮水机额定功率为 5W，按键后进入待机。",
    )
    unsafe = grounded_answer().model_copy(
        update={
            "summary": "CKIN PET F02 能够治疗疾病。",
            "sections": [
                GeneratedSection(
                    heading="错误生成",
                    content="该产品能够治疗疾病。",
                    source_ids=["S1"],
                )
            ],
        }
    )
    ollama = FakeOllama(unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="F02 待机功耗精确是多少瓦？"))

    assert response.status == "insufficient_evidence"
    assert response.evidence_status == "missing"
    assert response.sources == []
    assert response.decision.reason_code == "NO_DIRECT_EVIDENCE_FOR_REQUESTED_FACT"
    assert ollama.answer_calls == 0


async def test_realtime_inventory_requires_direct_inventory_evidence(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_capacity_fact(
        repository,
        tmp_path,
        suffix="inventory",
        content="CKIN PET F02 饮水机容量为 2.5L。",
    )
    ollama = FakeOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET F02 现在还有多少库存？"))

    assert response.status == "insufficient_evidence"
    assert response.evidence_status == "missing"
    assert response.decision.reason_code == "NO_DIRECT_EVIDENCE_FOR_REQUESTED_FACT"
    assert response.sources == []
    assert ollama.answer_calls == 0


def test_broad_business_question_gets_non_numeric_generation_guidance() -> None:
    factory_question = AgentService._generation_question("工厂主要做什么业务")
    company_question = AgentService._generation_question("公司主要做什么业务？")
    explicit_detail = AgentService._generation_question("公司主要业务和工厂面积是多少？")

    assert "不主动展开面积" in factory_question
    assert "不主动展开面积" in company_question
    assert explicit_detail == "公司主要业务和工厂面积是多少？"


def test_multi_model_retrieval_preserves_one_chunk_per_requested_model() -> None:
    question = "请比较 CKIN PET X01、F01 和 F02。"
    contents = (
        "CKIN PET X01 视频款支持远程查看。",
        "CKIN PET X01 按键款支持手动出粮。",
        "CKIN PET F01 视频款支持双频网络。",
        "CKIN PET F02 视频款支持防卡粮。",
    )
    chunks = [
        {
            "id": f"chunk-{index}",
            "document_id": "doc-models",
            "document_title": "型号资料",
            "filename": "models.txt",
            "page": 1,
            "sheet": None,
            "row_start": index,
            "row_end": index,
            "source_version": 1,
            "content": content,
            "embedding": pack_embedding([1.0, 0.0]),
        }
        for index, content in enumerate(contents, start=1)
    ]
    initial = retrieve(question, [1.0, 0.0], chunks, top_k=3)

    covered = AgentService._ensure_requested_model_coverage(
        question,
        [1.0, 0.0],
        chunks,
        initial,
        top_k=3,
    )

    combined = "\n".join(result.content for result in covered)
    assert all(code in combined for code in ("X01", "F01", "F02"))
    assert [result.source.source_id for result in covered] == ["S1", "S2", "S3"]


def test_extractive_fallback_keeps_specific_product_variant(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    question = "CKIN PET 双仓视频喂食器有哪些特点？"
    contents = (
        "双仓喂食器-视频款。双仓结构可分别储存宠粮，容量为 8L。",
        "CKIN PET 普通喂食器支持定时投喂。",
        "CKIN PET 智能饮水机支持循环出水。",
    )
    results = [
        SearchResult(
            source=SourceRef(
                source_id=f"S{index}",
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                document_title="产品资料",
                filename="products.txt",
                page=1,
                score=0.9,
                excerpt=content,
                source_version=1,
            ),
            content=content,
        )
        for index, content in enumerate(contents, start=1)
    ]

    fallback = service._extractive_fallback(  # noqa: SLF001
        question,
        service.boundary.classify(question),
        results,
        "UNSUPPORTED_SOURCE_MAPPING",
    )

    assert fallback is not None
    answer, guard = fallback
    assert guard.valid is True
    assert "双仓" in answer.summary
    assert "8L" in answer.summary
    assert "饮水机" not in answer.summary


@pytest.mark.parametrize(
    ("question", "history", "resolved_fragment", "reason_code"),
    [
        (
            "它的容量是多少？",
            [
                ChatMessage(role="user", content="请介绍 CKIN PET F02。"),
                ChatMessage(role="assistant", content="当前讨论的是 CKIN PET F02 饮水机。"),
            ],
            "F02",
            "OWNED_PRODUCT_ALLOWED",
        ),
        (
            "它怎么配网？",
            [
                ChatMessage(role="user", content="请介绍小佩喂食器。"),
                ChatMessage(role="assistant", content="我不能介绍第三方品牌产品。"),
            ],
            "小佩",
            "THIRD_PARTY_BRAND",
        ),
        (
            "它和小佩的喂食器哪个好？",
            [
                ChatMessage(role="user", content="请介绍 CKIN PET 智能视频喂食器。"),
                ChatMessage(role="assistant", content="当前讨论的是 CKIN PET 视频喂食器。"),
            ],
            "喂食器",
            "SELF_PRODUCT_ONLY_PARTIAL",
        ),
    ],
)
def test_history_resolves_only_recent_unique_product_subject(
    tmp_path: Path,
    question: str,
    history: list[ChatMessage],
    resolved_fragment: str,
    reason_code: str,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    request = ChatRequest(question=question, history=history)

    resolved = service.resolve_question(request)
    decision = service.boundary.classify(resolved)

    assert resolved_fragment in resolved
    assert decision.reason_code == reason_code


def test_history_with_multiple_models_keeps_demonstrative_ambiguous(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    request = ChatRequest(
        question="它怎么清洁？",
        history=[
            ChatMessage(role="user", content="请介绍 CKIN PET F01、F02 和 X01。"),
            ChatMessage(role="assistant", content="已介绍三个型号。"),
        ],
    )

    resolved = service.resolve_question(request)
    decision = service.boundary.classify(resolved)

    assert resolved == request.question
    assert decision.reason_code == "PRODUCT_IDENTITY_AMBIGUOUS"


@pytest.mark.parametrize("follow_up", ["它喜欢干什么", "它喜欢做什么"])
async def test_pet_pronoun_uses_recent_user_subject_and_general_model(
    tmp_path: Path, follow_up: str
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(general_answer=general_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question=follow_up,
        history=[
            ChatMessage(role="user", content="金毛喜欢吃什么"),
            ChatMessage(role="assistant", content="金毛日常饮食应以营养均衡为原则。"),
        ],
    )

    resolved = service.resolve_question(request)
    response = await service.chat(request)

    assert resolved == f"金毛{follow_up[1:]}"
    assert response.status == "general_knowledge"
    assert response.evidence_status == "model_knowledge"
    assert response.decision.reason_code == "PET_CARE_GENERAL_MODEL_KNOWLEDGE"
    assert response.sources == []
    assert ollama.topic_calls == 1
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0
    assert ollama.general_calls == 1


@pytest.mark.parametrize(
    ("follow_up", "resolved"),
    [
        ("喜欢和什么品种的同类玩", "金毛喜欢和什么品种的同类玩"),
        ("它喜欢和什么品种的同类玩", "金毛喜欢和什么品种的同类玩"),
        ("通常和哪些同伴一起互动？", "金毛通常和哪些同伴一起互动？"),
    ],
)
async def test_pet_social_follow_up_uses_recent_user_subject(
    tmp_path: Path, follow_up: str, resolved: str
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(general_answer=general_answer(), topic_kind="pet_care")
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question=follow_up,
        history=[
            ChatMessage(role="user", content="金毛喜欢吃什么"),
            ChatMessage(role="assistant", content="日常喂养建议营养均衡。"),
        ],
    )

    response = await service.chat(request)

    assert service.resolve_question(request) == resolved
    assert ollama.topic_questions == [resolved]
    assert response.status == "general_knowledge"
    assert response.decision.route_reason_code == "PET_CARE_GENERAL_MODEL_KNOWLEDGE"
    assert ollama.general_calls == 1


@pytest.mark.parametrize(
    "history",
    [
        [],
        [ChatMessage(role="assistant", content="它是金毛。")],
        [ChatMessage(role="user", content="金毛和柴犬喜欢吃什么？")],
        [ChatMessage(role="user", content="请介绍 CKIN PET F02 饮水机。")],
    ],
)
async def test_pet_social_follow_up_without_unique_user_pet_asks_for_subject(
    tmp_path: Path, history: list[ChatMessage]
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama()
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(question="喜欢和什么品种的同类玩", history=history)

    response = await service.chat(request)

    assert service.resolve_question(request) == request.question
    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"
    assert ollama.topic_calls == 0
    assert ollama.embed_calls == 0
    assert ollama.general_calls == 0


async def test_pet_social_follow_up_does_not_skip_intervening_company_topic(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama()
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="喜欢和什么品种的同类玩",
        history=[
            ChatMessage(role="user", content="金毛喜欢吃什么"),
            ChatMessage(role="assistant", content="饮食要均衡。"),
            ChatMessage(role="user", content="介绍一下公司"),
            ChatMessage(role="assistant", content="公司资料已介绍。"),
        ],
    )

    response = await service.chat(request)

    assert service.resolve_question(request) == request.question
    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"
    assert ollama.topic_calls == 0


@pytest.mark.parametrize(
    "follow_up",
    ["喜欢玩什么？", "通常喜欢什么运动？", "日常怎么训练？"],
)
async def test_elliptical_pet_follow_up_uses_recent_user_subject_when_model_confirms(
    tmp_path: Path, follow_up: str
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        general_answer=general_answer(),
        topic_kind="pet_care",
        use_prior_pet_subject=True,
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question=follow_up,
        history=[
            ChatMessage(role="user", content="金毛喜欢吃什么"),
            ChatMessage(role="assistant", content="日常喂养建议营养均衡。"),
        ],
    )

    response = await service.chat(request)

    assert service.resolve_question(request) == follow_up
    assert ollama.topic_questions == [follow_up]
    assert ollama.topic_prior_subjects == ["金毛"]
    assert ollama.general_questions == [f"金毛{follow_up}"]
    assert response.status == "general_knowledge"
    assert response.decision.reason_code == "PET_CARE_GENERAL_MODEL_KNOWLEDGE"


@pytest.mark.parametrize(
    "history",
    [
        [],
        [ChatMessage(role="assistant", content="它是金毛。")],
        [ChatMessage(role="user", content="金毛和柴犬喜欢吃什么？")],
    ],
)
async def test_elliptical_pet_follow_up_cannot_invent_missing_user_subject(
    tmp_path: Path, history: list[ChatMessage]
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(topic_kind="uncertain", use_prior_pet_subject=True)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="喜欢玩什么？", history=history))

    assert ollama.topic_prior_subjects == [None]
    assert ollama.general_calls == 0
    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"


async def test_elliptical_follow_up_accepts_user_named_breed_outside_policy_list(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        general_answer=general_answer(),
        topic_kind="pet_care",
        use_prior_pet_subject=True,
        referenced_pet_subject="秋田犬",
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="喜欢玩什么？",
        history=[
            ChatMessage(role="user", content="我养了一只秋田犬，喜欢吃什么？"),
            ChatMessage(role="assistant", content="狗狗一般需要均衡饮食。"),
        ],
    )

    response = await service.chat(request)

    assert response.status == "general_knowledge"
    assert ollama.topic_prior_subjects == [None]
    assert ollama.topic_recent_user_questions == [("我养了一只秋田犬，喜欢吃什么？",)]
    assert ollama.general_questions == ["秋田犬喜欢玩什么？"]


async def test_elliptical_follow_up_rejects_subject_absent_from_user_history(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        topic_kind="pet_care",
        use_prior_pet_subject=True,
        referenced_pet_subject="哈士奇",
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="喜欢玩什么？",
        history=[ChatMessage(role="user", content="金毛喜欢吃什么")],
    )

    response = await service.chat(request)

    assert ollama.topic_prior_subjects == ["金毛"]
    assert ollama.general_calls == 0
    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"


async def test_pet_context_survives_multiple_elliptical_user_turns(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        general_answer=general_answer(),
        topic_kind="pet_care",
        use_prior_pet_subject=True,
        referenced_pet_subject="秋田犬",
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="怎么训练？",
        history=[
            ChatMessage(role="user", content="我养了一只秋田犬，喜欢吃什么？"),
            ChatMessage(role="assistant", content="请提供均衡饮食。"),
            ChatMessage(role="user", content="喜欢玩什么？"),
            ChatMessage(role="assistant", content="可以选择安全的互动游戏。"),
        ],
    )

    response = await service.chat(request)

    assert response.status == "general_knowledge"
    assert ollama.topic_recent_user_questions == [
        ("我养了一只秋田犬，喜欢吃什么？", "喜欢玩什么？")
    ]
    assert ollama.general_questions == ["秋田犬怎么训练？"]


@pytest.mark.parametrize("intervening_question", ["介绍一下公司", "养花注意事项"])
async def test_pet_context_stops_at_explicit_topic_change(
    tmp_path: Path, intervening_question: str
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(topic_kind="uncertain")
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="喜欢玩什么？",
        history=[
            ChatMessage(role="user", content="金毛喜欢吃什么"),
            ChatMessage(role="assistant", content="请提供均衡饮食。"),
            ChatMessage(role="user", content=intervening_question),
            ChatMessage(role="assistant", content="已回答当前话题。"),
        ],
    )

    response = await service.chat(request)

    assert ollama.topic_recent_user_questions == [()]
    assert ollama.topic_prior_subjects == [None]
    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"


async def test_explicit_new_pet_subject_cannot_inherit_prior_breed(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        general_answer=general_answer(),
        topic_kind="pet_care",
        use_prior_pet_subject=True,
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="柴犬喜欢玩什么？",
        history=[ChatMessage(role="user", content="金毛喜欢吃什么")],
    )

    response = await service.chat(request)

    assert response.status == "general_knowledge"
    assert ollama.topic_prior_subjects == [None]
    assert ollama.general_questions == ["柴犬喜欢玩什么？"]


async def test_model_can_decline_to_carry_pet_subject_after_topic_switch(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(topic_kind="non_pet", use_prior_pet_subject=True)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    request = ChatRequest(
        question="养花注意事项",
        history=[ChatMessage(role="user", content="金毛喜欢吃什么")],
    )

    response = await service.chat(request)

    assert ollama.topic_prior_subjects == ["金毛"]
    assert ollama.general_calls == 0
    assert response.status == "out_of_scope"
    assert response.decision.reason_code == "UNRELATED_TOPIC"


def test_assistant_only_pet_claim_cannot_resolve_pronoun(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    request = ChatRequest(
        question="它喜欢干什么",
        history=[
            ChatMessage(role="user", content="它喜欢吃什么"),
            ChatMessage(role="assistant", content="它是金毛。"),
        ],
    )

    assert service.resolve_question(request) == request.question


@pytest.mark.parametrize(
    ("question", "expected_fragments"),
    [
        ("只比较容量", ("F02", "饮水机", "容量")),
        ("那容量呢？", ("F02", "饮水机", "容量")),
        ("WIFI款呢？", ("F02", "饮水机", "WIFI款", "容量")),
        ("它怎么设置？", ("F02", "饮水机", "设置")),
        ("那怎么清洁？", ("F02", "饮水机", "清洁")),
        ("再说说功能", ("F02", "饮水机", "功能")),
    ],
)
def test_elliptical_follow_up_preserves_unique_product_and_task(
    tmp_path: Path,
    question: str,
    expected_fragments: tuple[str, ...],
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    request = ChatRequest(
        question=question,
        history=[
            ChatMessage(role="user", content="CKIN PET F02 饮水机的容量是多少？"),
            ChatMessage(role="assistant", content="当前讨论的是 F02 饮水机。"),
        ],
    )

    resolved = service.resolve_question(request)

    assert all(fragment in resolved for fragment in expected_fragments)
    assert service.boundary.classify(resolved).reason_code == "OWNED_PRODUCT_ALLOWED"


@pytest.mark.parametrize(
    "question",
    [
        "猫咪平时怎么梳毛？",
        "猫咪呕吐该吃什么药？",
        "小佩喂食器呢？",
        "你好",
    ],
)
def test_topic_change_is_reclassified_without_previous_product(
    tmp_path: Path,
    question: str,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    request = ChatRequest(
        question=question,
        history=[
            ChatMessage(role="user", content="介绍 CKIN PET F02 饮水机。"),
            ChatMessage(role="assistant", content="当前讨论的是 F02 饮水机。"),
        ],
    )

    resolved = service.resolve_question(request)

    assert "F02" not in resolved


def test_mixed_medical_rewrite_preserves_exact_product_request(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        FakeOllama(),  # type: ignore[arg-type]
    )
    question = "请展示 CKIN PET F02 视频款喂食器参数，另外猫咪呕吐怎么办？"
    decision = service.boundary.classify(question)

    safe_question = service._self_only_question(question, decision)  # noqa: SLF001

    assert "CKIN PET" in safe_question
    assert "F02" in safe_question
    assert "视频款" in safe_question
    assert "喂食器" in safe_question
    assert "参数" in safe_question
    assert "呕吐" not in safe_question


async def test_withdrawal_during_generation_prevents_stale_answer(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", min_retrieval_score=0.0)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)

    class WithdrawingOllama(FakeOllama):
        async def answer(self, question: str, *_args: object, **_kwargs: object) -> GeneratedAnswer:
            result = await super().answer(question, *_args, **_kwargs)
            repository.withdraw_document("doc-1", actor_id="concurrency-test")
            return result

    ollama = WithdrawingOllama(grounded_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="CKIN PET 喂食器有什么功能？"))

    assert response.status == "insufficient_evidence"
    assert response.sources == []
    assert response.decision.reason_code == "SOURCE_CHANGED_DURING_GENERATION"


def test_metadata_change_is_draft_and_publish_validates_owner_and_time(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = KnowledgeService(settings, repository, FakeOllama())  # type: ignore[arg-type]

    draft = service.update_document("doc-1", {"title": "待审核标题"})
    assert draft["active_version"] == 1
    assert draft["source_version"] == 2
    assert draft["review_status"] == "pending"
    assert repository.fetch_searchable_chunks()[0]["document_title"] == "CKIN PET 产品资料"

    published = service.update_document(
        "doc-1", {"review_status": "approved", "publication_status": "published"}
    )
    assert published["active_version"] == 2
    assert repository.fetch_searchable_chunks()[0]["document_title"] == "待审核标题"

    with pytest.raises(ValueError, match="开始时间不能晚于结束时间"):
        service.update_document(
            "doc-1",
            {
                "effective_from": "2026-09-22T00:00:00+08:00",
                "effective_to": "2026-09-21T00:00:00+08:00",
            },
        )

    service.update_document("doc-1", {"owner_company_id": "third-party-company"})
    with pytest.raises(ValueError, match="归属于本公司"):
        service.update_document(
            "doc-1", {"review_status": "approved", "publication_status": "published"}
        )
    assert len(repository.fetch_searchable_chunks()) == 1


def test_publish_copies_media_to_immutable_version_directory(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = KnowledgeService(settings, repository, FakeOllama())  # type: ignore[arg-type]
    current_media_dir = settings.media_dir / "doc-1"
    current_media_dir.mkdir(parents=True)
    current_file = current_media_dir / "product.png"
    current_file.write_bytes(b"published-image-v1")
    repository.replace_media_assets(
        "doc-1",
        [
            {
                "id": "immutable-media",
                "image_key": "immutable-product",
                "filename": "product.png",
                "stored_path": str(current_file),
                "mime_type": "image/png",
                "sha256": "immutable-sha-v1",
                "width": 800,
                "height": 800,
                "source_kind": "product_image",
                "page": None,
                "sheet": "产品表",
                "row_start": 2,
                "row_end": 2,
                "cell_ref": "A2",
                "caption": "已发布图片",
                "alt_text": "已发布图片",
                "product_ids": ["P01"],
                "search_text": "P01 产品图",
            }
        ],
        mark_draft=True,
    )

    service.update_document(
        "doc-1", {"review_status": "approved", "publication_status": "published"}
    )
    public_asset = repository.get_public_media_asset("immutable-media")
    assert public_asset is not None
    published_file = Path(str(public_asset["stored_path"]))
    assert published_file.is_relative_to(settings.published_media_dir)
    assert published_file.read_bytes() == b"published-image-v1"

    current_file.write_bytes(b"unreviewed-image-v2")
    assert published_file.read_bytes() == b"published-image-v1"


async def test_boundary_response_skips_models_and_logs_only_question_hash(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama()
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    question = "猫咪呕吐该吃什么药？"

    response = await service.chat(ChatRequest(question=question))

    assert response.status == "out_of_scope"
    assert response.decision.reason_code == "PET_MEDICAL_ADVICE"
    assert response.fallback_kind == "policy_boundary"
    assert response.answer.refusal_reason is not None
    assert "专业兽医医疗范畴" in response.answer.refusal_reason
    assert response.sources == []
    assert ollama.topic_calls == 0
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0
    with repository.connect() as connection:
        row = connection.execute("SELECT question, fallback_kind FROM query_logs").fetchone()
    assert row is not None
    assert row["question"] != question
    assert len(row["question"]) == 64
    assert row["fallback_kind"] == "policy_boundary"


async def test_third_party_only_response_names_brand_and_explains_refusal(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama()
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="小佩的喂食器怎么样？"))

    assert response.status == "out_of_scope"
    assert "小佩并非本公司或旗下自有品牌产品" in response.answer.summary
    assert "不对小佩进行介绍、推荐、评价或比较" in response.answer.summary
    assert response.answer.refusal_reason is not None
    assert "无法保证评价的准确性和公正性" in response.answer.refusal_reason
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0


async def test_general_pet_care_uses_model_knowledge_without_documents(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(general_answer=general_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="猫砂盆多久清洁一次？"))

    assert response.status == "general_knowledge"
    assert response.evidence_status == "model_knowledge"
    assert response.sources == []
    assert response.metrics["retrieval_ms"] == 0
    assert ollama.embed_calls == 0
    assert ollama.general_calls == 1
    assert ollama.general_repair_calls == 0


async def test_slow_general_repair_uses_validated_local_salvage(tmp_path: Path) -> None:
    class SlowGeneralRepairOllama(FakeOllama):
        async def repair_general_knowledge(
            self, *_args: object, **_kwargs: object
        ) -> GeneralKnowledgeAnswer:
            self.general_repair_calls += 1
            await asyncio.Event().wait()
            raise AssertionError("repair should be cancelled")

    settings = Settings(data_dir=tmp_path / "data", chat_finalize_reserve_seconds=1)
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    unsafe = general_answer().model_copy(
        update={
            "summary": "日常及时清理结团和排泄物。每天使用84消毒液清洁猫砂盆。",
        }
    )
    ollama = SlowGeneralRepairOllama(general_answer=unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]
    deadline = asyncio.get_running_loop().time() + 1.5

    response = await service.chat(ChatRequest(question="猫砂盆多久清洁一次？"), deadline=deadline)

    assert response.status == "general_knowledge"
    assert response.metrics["repair_timed_out"] == 1
    assert "84" not in response.answer.summary
    assert ollama.general_repair_calls == 1
    assert asyncio.get_running_loop().time() < deadline


async def test_invalid_general_repair_uses_validated_local_salvage(tmp_path: Path) -> None:
    class InvalidGeneralRepairOllama(FakeOllama):
        async def repair_general_knowledge(
            self, *_args: object, **_kwargs: object
        ) -> GeneralKnowledgeAnswer:
            self.general_repair_calls += 1
            raise RuntimeError("模型没有返回合法的通用知识结构")

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    unsafe = general_answer().model_copy(
        update={
            "summary": "日常及时清理结团和排泄物。每天使用84消毒液清洁猫砂盆。",
        }
    )
    ollama = InvalidGeneralRepairOllama(general_answer=unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="猫砂盆日常应该如何清洁？"))

    assert response.status == "general_knowledge"
    assert response.evidence_status == "model_knowledge"
    assert response.metrics["repair_attempts"] == 1
    assert "84" not in response.answer.summary
    assert ollama.general_repair_calls == 1


async def test_general_repair_keeps_safe_clauses_when_every_sentence_claims_medical_benefit(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    unsafe = GeneralKnowledgeAnswer(
        title="宠物生活区通风清洁",
        summary="定期开窗通风，清理地面和用品，预防呼吸道疾病。",
        sections=[
            AnswerSection(
                heading="日常清理",
                content="及时清理排泄物和食物残渣，保持环境干燥，降低感染风险。",
            )
        ],
        follow_up_questions=[],
    )
    ollama = FakeOllama(general_answer=unsafe)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="宠物生活区怎样保持通风和清洁？"))

    assert response.status == "general_knowledge"
    assert response.evidence_status == "model_knowledge"
    assert response.metrics["repair_attempts"] == 1
    assert "定期开窗通风" in response.answer.summary
    assert "及时清理排泄物" in response.answer.sections[0].content
    assert "预防呼吸道疾病" not in response.answer.summary
    assert "降低感染风险" not in response.answer.sections[0].content
    assert ollama.general_repair_calls == 1


async def test_unavailable_general_repair_is_not_hidden_by_salvage(tmp_path: Path) -> None:
    class UnavailableGeneralRepairOllama(FakeOllama):
        async def repair_general_knowledge(
            self, *_args: object, **_kwargs: object
        ) -> GeneralKnowledgeAnswer:
            raise OllamaUnavailableError("模型连接中断")

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    unsafe = general_answer().model_copy(
        update={"summary": "日常清理结团。每天使用84消毒液清洁猫砂盆。"}
    )
    service = AgentService(
        settings,
        repository,
        UnavailableGeneralRepairOllama(general_answer=unsafe),  # type: ignore[arg-type]
    )

    response = await service.chat(ChatRequest(question="猫砂盆日常应该如何清洁？"))

    assert response.status == "service_unavailable"


async def test_breed_habit_question_uses_general_knowledge_instead_of_company_rag(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(general_answer=general_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="金毛的习惯"))

    assert response.status == "general_knowledge"
    assert response.evidence_status == "model_knowledge"
    assert response.decision.reason_code == "PET_CARE_GENERAL_MODEL_KNOWLEDGE"
    assert response.sources == []
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0
    assert ollama.general_calls == 1


async def test_westie_care_question_uses_general_knowledge_without_company_retrieval(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(
        general_answer=GeneralKnowledgeAnswer(
            title="西高地日常养护",
            summary="西高地犬需要规律活动、梳理毛发和稳定的日常作息。",
            sections=[
                AnswerSection(
                    heading="日常安排",
                    content="结合年龄和个体状态安排散步、互动与日常梳毛。",
                )
            ],
        )
    )
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="小白养西高地注意事项"))

    assert response.status == "general_knowledge"
    assert response.decision.reason_code == "PET_CARE_GENERAL_MODEL_KNOWLEDGE"
    assert response.sources == []
    assert response.metrics["retrieval_ms"] == 0
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0
    assert ollama.general_calls == 1


@pytest.mark.parametrize(
    ("question", "kind", "expected_status", "expected_reason", "expected_general_calls"),
    [
        (
            "新手养松狮注意事项",
            "pet_care",
            "general_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            1,
        ),
        (
            "新手养秋田注意事项",
            "pet_care",
            "general_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            1,
        ),
        (
            "秋田喜欢吃什么",
            "pet_care",
            "general_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            1,
        ),
        ("养花注意事项", "non_pet", "out_of_scope", "UNRELATED_TOPIC", 0),
        ("狗多少钱", "pet_other", "out_of_scope", "UNRELATED_TOPIC", 0),
        (
            "养它注意事项",
            "uncertain",
            "clarification_needed",
            "PET_SUBJECT_AMBIGUOUS",
            0,
        ),
    ],
)
async def test_unlisted_care_subject_is_classified_before_answering(
    tmp_path: Path,
    question: str,
    kind: Literal["pet_care", "pet_other", "non_pet", "uncertain"],
    expected_status: str,
    expected_reason: str,
    expected_general_calls: int,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = FakeOllama(general_answer=general_answer(), topic_kind=kind)
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question=question))

    assert response.status == expected_status
    assert response.decision.reason_code == expected_reason
    assert response.sources == []
    assert response.metrics["topic_check_ms"] >= 0
    assert ollama.topic_calls == 1
    assert ollama.general_calls == expected_general_calls
    assert ollama.embed_calls == 0
    assert ollama.answer_calls == 0


async def test_invalid_pet_care_classification_asks_for_clarification(tmp_path: Path) -> None:
    class InvalidTopicOllama(FakeOllama):
        async def classify_question_topic(
            self,
            _question: str,
            _company_name: str,
            _owned_aliases: tuple[str, ...],
            prior_pet_subject: str | None = None,
            recent_user_questions: tuple[str, ...] = (),
        ) -> QuestionTopicClassification:
            del prior_pet_subject, recent_user_questions
            self.topic_calls += 1
            raise RuntimeError("invalid structured topic")

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    ollama = InvalidTopicOllama(general_answer=general_answer())
    service = AgentService(settings, repository, ollama)  # type: ignore[arg-type]

    response = await service.chat(ChatRequest(question="新手养陌生名字注意事项"))

    assert response.status == "clarification_needed"
    assert response.decision.reason_code == "PET_SUBJECT_AMBIGUOUS"
    assert ollama.topic_calls == 1
    assert ollama.general_calls == 0
    assert ollama.embed_calls == 0


def test_general_pet_care_documents_cannot_be_assigned(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    service = KnowledgeService(settings, repository, FakeOllama())  # type: ignore[arg-type]

    try:
        service.update_document("doc-1", {"knowledge_domain": "pet_care_general"})
    except ValueError as exc:
        assert "不接受文档上传" in str(exc)
    else:
        raise AssertionError("通用养宠知识不应接受文档知识域")


async def test_reindex_preserves_manual_content_and_enabled_state(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    seed_published_product(repository, tmp_path)
    repository.update_chunk("chunk-1", content="运营手工修订内容", enabled=False)
    ollama = FakeOllama()
    service = KnowledgeService(settings, repository, ollama)  # type: ignore[arg-type]

    await service.reindex_document("doc-1")
    chunk = repository.get_chunk("chunk-1")

    assert chunk is not None
    assert chunk["content"] == "运营手工修订内容"
    assert chunk["enabled"] == 0


async def test_document_delete_moves_source_to_recoverable_trash(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare_directories()
    repository = KnowledgeRepository(settings.database_path)
    repository.initialize()
    document_dir = settings.document_dir / "doc-1"
    document_dir.mkdir()
    stored_path = document_dir / "facts.md"
    stored_path.write_text("公司事实", encoding="utf-8")
    now = utc_now()
    repository.create_document(
        {
            "id": "doc-1",
            "title": "公司事实",
            "filename": "facts.md",
            "stored_path": str(stored_path),
            "file_type": "md",
            "size_bytes": 12,
            "sha256": "delete-test",
            "status": "READY",
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    service = KnowledgeService(settings, repository, FakeOllama())  # type: ignore[arg-type]

    assert await service.delete_document("doc-1") is True
    assert repository.get_document("doc-1") is None
    trash_items = list(settings.trash_dir.iterdir())
    assert len(trash_items) == 1
    assert (trash_items[0] / "facts.md").read_text(encoding="utf-8") == "公司事实"
