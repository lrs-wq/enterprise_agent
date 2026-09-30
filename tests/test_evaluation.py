import csv
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from company_agent.config import Settings
from company_agent.database import KnowledgeRepository, pack_embedding, utc_now
from company_agent.evaluation import (
    EvaluationCase,
    EvaluationResult,
    QAEvaluator,
    apply_baseline,
    clone_database,
    load_cases,
    load_checkpoint,
    write_results,
)
from company_agent.schemas import (
    AnswerSection,
    ChatMessage,
    GeneralKnowledgeAnswer,
    GeneratedAnswer,
    GeneratedSection,
    QuestionTopicClassification,
)


class EvaluationFakeModel:
    def __init__(self) -> None:
        self.answer_histories: list[list[ChatMessage]] = []
        self.topic_questions: list[str] = []
        self.topic_prior_subjects: list[str | None] = []
        self.topic_recent_user_questions: list[tuple[str, ...]] = []
        self.general_questions: list[str] = []

    async def classify_question_topic(
        self,
        question: str,
        _company_name: str,
        _owned_aliases: Sequence[str],
        prior_pet_subject: str | None = None,
        recent_user_questions: Sequence[str] = (),
    ) -> QuestionTopicClassification:
        self.topic_questions.append(question)
        self.topic_prior_subjects.append(prior_pet_subject)
        self.topic_recent_user_questions.append(tuple(recent_user_questions))
        if "同类" in question:
            return QuestionTopicClassification(kind="pet_care")
        if question == "喜欢玩什么？" and recent_user_questions:
            subject = "秋田犬" if "秋田犬" in recent_user_questions[-1] else "金毛"
            return QuestionTopicClassification(
                kind="pet_care",
                use_prior_pet_subject=True,
                referenced_pet_subject=subject,
            )
        if question == "喜欢玩什么？":
            return QuestionTopicClassification(kind="uncertain")
        if question == "养花注意事项":
            return QuestionTopicClassification(kind="non_pet")
        if question == "柴犬喜欢玩什么？":
            return QuestionTopicClassification(kind="pet_care")
        kind = "company" if "公司" in question and "喂食器" not in question else "pet_product"
        return QuestionTopicClassification(kind=kind)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def answer(
        self,
        _question: str,
        _context: str,
        history: Sequence[ChatMessage],
        _allowed_source_ids: Sequence[str],
    ) -> GeneratedAnswer:
        self.answer_histories.append(list(history))
        return GeneratedAnswer(
            status="grounded",
            intent="product_intro",
            title="测试产品",
            summary="CKIN PET 测试产品支持定时定量投喂。",
            summary_source_ids=["S1"],
            sections=[
                GeneratedSection(
                    heading="功能",
                    content="测试产品支持定时定量投喂。",
                    source_ids=["S1"],
                )
            ],
            follow_up_questions=[],
        )

    async def repair_answer(
        self,
        question: str,
        context: str,
        previous: GeneratedAnswer,
        allowed_source_ids: Sequence[str],
        validation_error: str,
        validation_detail: str | None = None,
    ) -> GeneratedAnswer:
        del question, context, allowed_source_ids, validation_error, validation_detail
        return previous

    async def answer_general_knowledge(
        self, _question: str, _history: Sequence[ChatMessage]
    ) -> GeneralKnowledgeAnswer:
        self.general_questions.append(_question)
        return GeneralKnowledgeAnswer(
            title="日常养宠",
            summary="保持环境清洁并观察宠物状态。",
            sections=[AnswerSection(heading="建议", content="定期清洁用品。")],
            follow_up_questions=[],
        )

    async def repair_general_knowledge(
        self,
        _question: str,
        previous: GeneralKnowledgeAnswer,
        _validation_error: str,
    ) -> GeneralKnowledgeAnswer:
        return previous


def make_case(
    case_id: str,
    question: str,
    *,
    status: str,
    evidence: str,
    reason: str,
) -> EvaluationCase:
    row = {
        "用例ID": case_id,
        "问答分类": "测试",
        "测试场景": "测试场景",
        "优先级": "P0",
        "前置/对话上下文": "测试前置",
        "用户询问语句": question,
        "预期回答状态": status,
        "预期证据状态": evidence,
        "预期原因码": reason,
        "预期模型回答结果形式": "",
        "回答必须包含": "",
        "回答禁止包含": "",
        "实际模型回答": "",
        "实际状态": "",
        "测试结果": "",
        "备注": "",
    }
    return EvaluationCase.from_row(row)


def seed_source_database(path: Path) -> None:
    repository = KnowledgeRepository(path)
    repository.initialize()
    now = utc_now()
    repository.create_document(
        {
            "id": "source-doc",
            "title": "CKIN PET 测试资料",
            "filename": "source.txt",
            "stored_path": "/source/source.txt",
            "file_type": "txt",
            "size_bytes": 100,
            "sha256": "evaluation-source",
            "status": "READY",
            "knowledge_domain": "product_public",
            "created_at": now,
            "updated_at": now,
        }
    )
    repository.replace_chunks(
        "source-doc",
        [
            {
                "id": "source-chunk",
                "position": 0,
                "content": "CKIN PET 测试产品支持定时定量投喂。",
                "page": 1,
                "sheet": None,
                "row_start": None,
                "row_end": None,
                "embedding": pack_embedding([1.0, 0.0]),
            }
        ],
    )
    repository.publish_document("source-doc", reviewer_id="tester")


@pytest.mark.parametrize(
    ("case_id", "question", "state_key"),
    [
        ("QA-237", "请介绍一下公司。", "empty"),
        ("QA-238", "请介绍草稿文档里提到的新喂食器。", "draft"),
        ("QA-239", "被撤回宣传册中的 F02 参数是什么？", "withdrawn"),
        ("QA-240", "旧资料里写的公司面积现在是多少？", "expired"),
    ],
)
async def test_lifecycle_preconditions_are_really_applied(
    tmp_path: Path,
    case_id: str,
    question: str,
    state_key: str,
) -> None:
    source = tmp_path / "source.db"
    seed_source_database(source)
    settings = Settings(data_dir=tmp_path / "unused", min_retrieval_score=0.0)
    evaluator = QAEvaluator(settings, source, EvaluationFakeModel())
    case = make_case(
        case_id,
        question,
        status="insufficient_evidence",
        evidence="missing",
        reason=(
            "COMPANY_PUBLIC_ALLOWED" if case_id in {"QA-237", "QA-240"} else "OWNED_PRODUCT_ALLOWED"
        ),
    )

    result = await evaluator.evaluate(case)

    assert result.passed
    assert result.precondition_state["database_fixture"] == state_key
    searched_domain = "company" if case_id in {"QA-237", "QA-240"} else "product"
    assert result.precondition_state[f"searchable_{searched_domain}_chunks"] == 0


async def test_pet_social_follow_up_case_resolves_user_subject(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    seed_source_database(source)
    settings = Settings(data_dir=tmp_path / "unused")
    model = EvaluationFakeModel()
    evaluator = QAEvaluator(settings, source, model)
    case = make_case(
        "QA-266",
        "喜欢和什么品种的同类玩",
        status="general_knowledge",
        evidence="model_knowledge",
        reason="PET_CARE_GENERAL_MODEL_KNOWLEDGE",
    )

    result = await evaluator.evaluate(case)

    assert result.passed
    assert model.topic_questions == ["金毛喜欢和什么品种的同类玩"]


@pytest.mark.parametrize(
    ("case_id", "question", "status", "evidence", "reason", "prior", "generated"),
    [
        (
            "QA-268",
            "喜欢玩什么？",
            "general_knowledge",
            "model_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            "金毛",
            "金毛喜欢玩什么？",
        ),
        (
            "QA-269",
            "养花注意事项",
            "out_of_scope",
            "not_applicable",
            "UNRELATED_TOPIC",
            "金毛",
            None,
        ),
        (
            "QA-270",
            "柴犬喜欢玩什么？",
            "general_knowledge",
            "model_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            None,
            "柴犬喜欢玩什么？",
        ),
        (
            "QA-271",
            "喜欢玩什么？",
            "clarification_needed",
            "not_applicable",
            "PET_SUBJECT_AMBIGUOUS",
            None,
            None,
        ),
        (
            "QA-272",
            "喜欢玩什么？",
            "general_knowledge",
            "model_knowledge",
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            None,
            "秋田犬喜欢玩什么？",
        ),
    ],
)
async def test_elliptical_pet_follow_up_matrix_cases(
    tmp_path: Path,
    case_id: str,
    question: str,
    status: str,
    evidence: str,
    reason: str,
    prior: str | None,
    generated: str | None,
) -> None:
    source = tmp_path / "source.db"
    seed_source_database(source)
    model = EvaluationFakeModel()
    evaluator = QAEvaluator(Settings(data_dir=tmp_path / "unused"), source, model)
    case = make_case(case_id, question, status=status, evidence=evidence, reason=reason)

    result = await evaluator.evaluate(case)

    assert result.passed
    assert model.topic_questions == [question]
    assert model.topic_prior_subjects == [prior]
    assert model.general_questions == ([] if generated is None else [generated])


async def test_each_case_uses_private_database_and_does_not_mutate_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    seed_source_database(source)
    settings = Settings(data_dir=tmp_path / "unused", min_retrieval_score=0.0)
    evaluator = QAEvaluator(settings, source, EvaluationFakeModel())
    empty_case = make_case(
        "QA-237",
        "请介绍一下公司。",
        status="insufficient_evidence",
        evidence="missing",
        reason="COMPANY_PUBLIC_ALLOWED",
    )
    normal_case = make_case(
        "QA-X",
        "CKIN PET 测试产品有什么功能？",
        status="grounded",
        evidence="supported",
        reason="OWNED_PRODUCT_ALLOWED",
    )

    empty_result = await evaluator.evaluate(empty_case)
    normal_result = await evaluator.evaluate(normal_case)

    assert empty_result.passed
    assert normal_result.passed
    repository = KnowledgeRepository(source)
    assert len(repository.fetch_searchable_chunks(("product_public",))) == 1
    with repository.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0] == 0


async def test_history_fixture_is_sent_to_the_service(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    seed_source_database(source)
    model = EvaluationFakeModel()
    settings = Settings(data_dir=tmp_path / "unused", min_retrieval_score=0.0)
    evaluator = QAEvaluator(settings, source, model)
    case = make_case(
        "QA-262",
        "谢谢，CKIN PET F02 有哪些功能？",
        status="grounded",
        evidence="supported",
        reason="OWNED_PRODUCT_ALLOWED",
    )

    result = await evaluator.evaluate(case)

    assert result.passed
    assert result.precondition_state["history_messages"] == 2
    assert [message.role for message in model.answer_histories[0]] == ["user", "assistant"]
    assert "F02" in model.answer_histories[0][0].content


async def test_fixed_boundary_case_does_not_call_model_or_source_database(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing-source.db"
    settings = Settings(data_dir=tmp_path / "unused")
    evaluator = QAEvaluator(settings, source, EvaluationFakeModel())
    case = make_case(
        "QA-248",
        "介绍一下小佩公司。",
        status="out_of_scope",
        evidence="not_applicable",
        reason="THIRD_PARTY_BRAND",
    )

    result = await evaluator.evaluate(case)

    assert result.passed
    assert result.precondition_state["searchable_company_chunks"] == 0
    assert result.precondition_state["searchable_product_chunks"] == 0


def test_clone_database_is_independent(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    clone = tmp_path / "clone.db"
    seed_source_database(source)

    clone_database(source, clone)
    clone_repository = KnowledgeRepository(clone)
    clone_repository.withdraw_document("source-doc", actor_id="test")

    source_repository = KnowledgeRepository(source)
    assert len(source_repository.fetch_searchable_chunks(("product_public",))) == 1
    assert clone_repository.fetch_searchable_chunks(("product_public",)) == []


def test_result_csv_contains_diagnostics_and_regression(tmp_path: Path) -> None:
    case = make_case(
        "QA-X",
        "问题",
        status="grounded",
        evidence="supported",
        reason="OWNED_PRODUCT_ALLOWED",
    )
    result = EvaluationResult(
        case=case,
        actual_answer=json.dumps({"answer": "ok"}, ensure_ascii=False),
        actual_status="insufficient_evidence",
        actual_evidence_status="missing",
        actual_reason_code="OWNED_PRODUCT_ALLOWED",
        actual_result_reason_code="LOW_RETRIEVAL_RELEVANCE",
        failure_stage="检索",
        failure_field="searchable_chunks",
        execution_error="",
        duration_ms=12,
        precondition_state={"database_fixture": "snapshot"},
        passed=False,
    )
    apply_baseline([result], {"QA-X": "通过"})
    output = tmp_path / "result.csv"

    write_results(output, [result])

    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["实际证据状态"] == "missing"
    assert row["实际原因码"] == "OWNED_PRODUCT_ALLOWED"
    assert row["实际路由原因码"] == "OWNED_PRODUCT_ALLOWED"
    assert row["实际结果原因码"] == "LOW_RETRIEVAL_RELEVANCE"
    assert row["实际失败阶段"] == "检索"
    assert row["实际失败字段"] == "searchable_chunks"
    assert row["回归结果"] == "新增异常"


def test_load_cases_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv"
    rows = [
        make_case("QA-X", "问题", status="grounded", evidence="supported", reason="ROUTE").raw
    ] * 2
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="必须唯一"):
        load_cases(path)


def test_checkpoint_resume_requires_same_case_prefix(tmp_path: Path) -> None:
    first = make_case("QA-X", "第一问", status="grounded", evidence="supported", reason="ROUTE")
    second = make_case("QA-Y", "第二问", status="grounded", evidence="supported", reason="ROUTE")
    result = EvaluationResult(
        case=first,
        actual_answer="回答",
        actual_status="grounded",
        actual_evidence_status="supported",
        actual_reason_code="ROUTE",
        actual_result_reason_code="ROUTE",
        failure_stage="",
        failure_field="",
        execution_error="",
        duration_ms=10,
        precondition_state={"ready": True},
        passed=True,
    )
    path = tmp_path / "checkpoint.csv"
    write_results(path, [result])

    restored = load_checkpoint(path, [first, second])
    assert len(restored) == 1
    assert restored[0].case is first
    assert restored[0].actual_answer == "回答"
    assert restored[0].precondition_state == {"ready": True}

    changed_first = make_case(
        "QA-X", "已修改的第一问", status="grounded", evidence="supported", reason="ROUTE"
    )
    with pytest.raises(ValueError, match="不一致"):
        load_checkpoint(path, [changed_first, second])
