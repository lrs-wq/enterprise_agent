import pytest

from company_agent.answer_guard import (
    salvage_general_knowledge_answer,
    salvage_supported_answer,
    validate_general_knowledge_answer,
    validate_generated_answer,
)
from company_agent.retrieval import SearchResult
from company_agent.schemas import (
    AnswerSection,
    GeneralKnowledgeAnswer,
    GeneratedAnswer,
    GeneratedSection,
    SourceRef,
)


def result(content: str, source_id: str = "S1") -> SearchResult:
    return SearchResult(
        source=SourceRef(
            source_id=source_id,
            chunk_id=f"chunk-{source_id}",
            document_id="doc-1",
            document_title="产品手册",
            filename="manual.pdf",
            page=2,
            score=0.9,
            excerpt=content,
        ),
        content=content,
    )


def generated(summary: str, section: str, cited: list[str]) -> GeneratedAnswer:
    return GeneratedAnswer(
        status="grounded",
        intent="product_intro",
        title="视频款喂食器",
        summary=summary,
        summary_source_ids=cited,
        sections=[GeneratedSection(heading="功能", content=section, source_ids=cited)],
        follow_up_questions=[],
    )


def test_guard_accepts_structured_supported_citations() -> None:
    evidence = result("视频款喂食器带 110° 广角夜视摄像头。")
    answer = generated(
        "视频款喂食器带 110° 广角夜视摄像头。",
        "摄像头为 110° 广角并支持夜视。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True
    assert checked.cited_source_ids == ("S1",)


def test_guard_ignores_non_factual_evidence_preamble() -> None:
    evidence = result("该设备支持定时定量投喂。")
    answer = generated(
        "根据知识库，该设备支持定时定量投喂。",
        "该设备支持定时定量投喂。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True


def test_guard_rejects_invented_source_id() -> None:
    evidence = result("视频款喂食器带摄像头。")
    answer = generated(
        "视频款喂食器带摄像头。",
        "视频款带摄像头。",
        ["S99"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNKNOWN_SOURCE_ID"


def test_guard_rejects_unsupported_measured_value() -> None:
    evidence = result("视频款喂食器带 110° 广角夜视摄像头。")
    answer = generated(
        "视频款喂食器带 120° 广角夜视摄像头。",
        "摄像头为 120° 广角。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_MEASURED_VALUE"


def test_guard_rejects_unsupported_voltage_unit_value() -> None:
    evidence = result("输入电压为 5V。")
    answer = generated("输入电压为 24V。", "设备使用 24V 输入电压。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_MEASURED_VALUE"


def test_guard_rejects_reversed_numeric_condition() -> None:
    evidence = result("设备功率不超过 5W。")
    answer = generated("设备功率至少为 5W。", "设备功率不低于 5W。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_VALUE_CONDITION"


def test_guard_normalizes_colloquial_area_and_exact_to_approximate_claim() -> None:
    evidence = result("工厂整体面积48000余平。产业园区土地面积135.92亩。")
    answer = generated(
        "工厂整体面积约48000平方米。",
        "产业园区占地面积约135.92亩。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True


def test_guard_normalizes_ocr_area_unit_and_chinese_counts() -> None:
    evidence = result("新建厂房2.3万m，布局四大中心，旗下有三大品牌。")
    answer = generated(
        "新建厂房面积2.3万平方米。",
        "公司布局4大中心，旗下有3大品牌。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True


def test_guard_still_rejects_total_area_rebound_to_workshop() -> None:
    evidence = result("工厂整体面积48000余平，其中生产车间33000平，办公室15000平。")
    answer = generated(
        "生产车间约48000平方米。",
        "工厂拥有生产车间。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_ATTRIBUTE_VALUE_BINDING"


def test_guard_salvages_supported_claims_after_failed_repair() -> None:
    evidence = result(
        "公司聚焦宠物用品研发、生产与国内外销售。"
        "工厂制造工艺涵盖吹塑、注塑。"
        "工厂整体面积48000余平，其中生产车间33000平。"
    )
    answer = generated(
        "公司聚焦宠物用品研发、生产与国内外销售。生产车间约48000平方米。",
        "工厂制造工艺涵盖吹塑、注塑。",
        ["S1"],
    )

    salvaged = salvage_supported_answer(answer, [evidence])

    assert salvaged is not None
    candidate, checked, removed_count = salvaged
    assert checked.valid is True
    assert removed_count == 1
    assert "48000" not in candidate.summary
    assert "研发、生产与国内外销售" in candidate.summary


@pytest.mark.parametrize(
    ("before_claim", "after_claim", "expected_separator"),
    [
        ("。", "，", "。"),
        ("，", "。", "。"),
        ("，", "，", "，"),
        ("。", "。", "。"),
        ("。 ", "， ", "。"),
    ],
)
def test_guard_salvage_rejoins_punctuation_around_removed_claim(
    before_claim: str, after_claim: str, expected_separator: str
) -> None:
    evidence = result("CKIN PET 智能喂食器支持定时定量投喂。该设备支持手动出粮。")
    answer = generated(
        "CKIN PET 智能喂食器支持定时定量投喂。",
        (
            f"CKIN PET 智能喂食器支持定时定量投喂{before_claim}"
            f"该设备的容量为24L{after_claim}该设备支持手动出粮。"
        ),
        ["S1"],
    )

    salvaged = salvage_supported_answer(answer, [evidence])

    assert salvaged is not None
    candidate, checked, removed_count = salvaged
    assert checked.valid is True
    assert removed_count == 1
    assert candidate.sections[0].content == (
        f"CKIN PET 智能喂食器支持定时定量投喂{expected_separator}该设备支持手动出粮。"
    )


def test_guard_salvage_rejoins_punctuation_after_consecutive_removals() -> None:
    evidence = result("CKIN PET 智能喂食器支持定时定量投喂。该设备支持手动出粮。")
    answer = generated(
        "CKIN PET 智能喂食器支持定时定量投喂。",
        (
            "CKIN PET 智能喂食器支持定时定量投喂。"
            "该设备的容量为24L，该设备的功率为999W，该设备支持手动出粮。"
        ),
        ["S1"],
    )

    salvaged = salvage_supported_answer(answer, [evidence])

    assert salvaged is not None
    candidate, checked, removed_count = salvaged
    assert checked.valid is True
    assert removed_count == 2
    assert candidate.sections[0].content == (
        "CKIN PET 智能喂食器支持定时定量投喂。该设备支持手动出粮。"
    )


def test_guard_salvage_removes_only_redundant_source_mapping() -> None:
    supporting = result("公司聚焦宠物用品研发、生产与销售。", "S1")
    unrelated = result("产业园设有新能源零碳示范区。", "S2")
    answer = generated(
        "公司聚焦宠物用品研发、生产与销售。[S1][S2]",
        "公司的业务涵盖宠物用品研发、生产与销售。[S1][S2]",
        ["S1", "S2"],
    )

    initial = validate_generated_answer(answer, [supporting, unrelated])
    salvaged = salvage_supported_answer(answer, [supporting, unrelated])

    assert initial.reason == "UNSUPPORTED_SOURCE_MAPPING"
    assert salvaged is not None
    candidate, checked, repaired_count = salvaged
    assert checked.valid is True
    assert repaired_count == 2
    assert candidate.summary_source_ids == ["S1"]
    assert candidate.sections[0].source_ids == ["S1"]
    assert "[S2]" not in candidate.summary
    assert "[S2]" not in candidate.sections[0].content


def test_guard_salvage_does_not_keep_claim_with_only_unrelated_source() -> None:
    unrelated = result("产业园设有新能源零碳示范区。")
    answer = generated(
        "公司聚焦宠物用品研发、生产与销售。",
        "公司的业务涵盖宠物用品研发、生产与销售。",
        ["S1"],
    )

    assert salvage_supported_answer(answer, [unrelated]) is None


def test_guard_rejects_reversed_support_polarity() -> None:
    evidence = result("该设备不支持离线定时。")
    answer = generated("该设备支持离线定时。", "设备支持离线定时。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "CONTRADICTED_CLAIM_POLARITY"
    assert checked.repair_detail is not None
    assert "支持离线定时" in checked.repair_detail


def test_guard_rejects_value_copied_from_another_model() -> None:
    evidence = result("型号 A 容量 2L，型号 B 容量 5L。")
    answer = generated("型号 A 容量为 5L。", "型号 A 提供 5L 容量。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_SUBJECT_VALUE_BINDING"


def test_guard_accepts_value_bound_to_the_same_model() -> None:
    evidence = result("型号 A 容量 2L，型号 B 容量 5L。")
    answer = generated("型号 A 容量为 2L。", "型号 A 提供 2L 容量。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True


def test_guard_does_not_treat_numbered_list_markers_as_product_values() -> None:
    evidence = result("产品包括喂食器、饮水机和烘干机。")
    answer = generated(
        "产品包括喂食器、饮水机和烘干机。",
        "产品包括：<br>1. 喂食器<br>2. 饮水机<br>3. 烘干机",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True


def test_guard_rejects_universal_claim_supported_only_for_some_models() -> None:
    evidence = result("部分型号支持离线定时。")
    answer = generated("所有型号都支持离线定时。", "全系均支持离线定时。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "CONTRADICTED_CLAIM_QUALIFIER"


def test_guard_rejects_standard_feature_when_evidence_says_optional() -> None:
    evidence = result("摄像头为选配功能。")
    answer = generated("摄像头是标配功能。", "设备标配摄像头。", ["S1"])

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "CONTRADICTED_CLAIM_QUALIFIER"


def test_guard_rejects_unsupported_independent_claim_in_same_block() -> None:
    evidence = result("该设备支持定时。")
    answer = generated(
        "该设备支持定时。它还获得了国际设计大奖。",
        "该设备支持定时。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_CLAIM"


def test_guard_rejects_inline_source_not_in_structured_mapping() -> None:
    evidence = result("视频款喂食器带摄像头。")
    answer = generated(
        "视频款喂食器带摄像头。[S99]",
        "视频款带摄像头。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "INLINE_SOURCE_MISMATCH"


def test_guard_rejects_third_party_content_after_generation() -> None:
    evidence = result("视频款喂食器带摄像头。")
    answer = generated(
        "视频款喂食器带摄像头，比小佩更好。",
        "视频款带摄像头。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence], ("小佩",))

    assert checked.valid is False
    assert checked.reason == "FORBIDDEN_THIRD_PARTY_CONTENT"


def test_guard_rejects_third_party_content_in_follow_up() -> None:
    evidence = result("视频款喂食器带摄像头。")
    answer = generated(
        "视频款喂食器带摄像头。",
        "视频款带摄像头。",
        ["S1"],
    ).model_copy(update={"follow_up_questions": ["要不要继续比较小佩？"]})

    checked = validate_generated_answer(answer, [evidence], ("小佩",))

    assert checked.valid is False
    assert checked.reason == "FORBIDDEN_THIRD_PARTY_CONTENT"


def test_guard_rejects_third_party_content_in_section_heading() -> None:
    evidence = result("视频款喂食器带摄像头。")
    answer = generated(
        "视频款喂食器带摄像头。",
        "视频款带摄像头。",
        ["S1"],
    )
    answer.sections[0].heading = "小佩产品推荐"

    checked = validate_generated_answer(answer, [evidence], ("小佩",))

    assert checked.valid is False
    assert checked.reason == "FORBIDDEN_THIRD_PARTY_CONTENT"


def test_guard_rejects_unverified_brand_appended_to_body() -> None:
    evidence = result("该设备支持定时。")
    answer = generated(
        "该设备支持定时。",
        "该设备支持定时。未登记品牌甲值得推荐。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNVERIFIED_THIRD_PARTY_CONTENT"


def test_guard_checks_measured_value_in_section_heading() -> None:
    evidence = result("输入电压为 5V。")
    answer = generated("输入电压为 5V。", "设备采用 5V 输入。", ["S1"])
    answer.sections[0].heading = "24V 供电"

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_MEASURED_VALUE"


def test_guard_checks_measured_value_in_follow_up_question() -> None:
    evidence = result("输入电压为 5V。")
    answer = generated("输入电压为 5V。", "设备采用 5V 输入。", ["S1"])
    answer.follow_up_questions = ["是否还想了解 24V 输入电压版本？"]

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_MEASURED_VALUE"


def test_guard_does_not_bind_rated_power_to_standby_power() -> None:
    evidence = result("CKIN PET F02 的额定功率为 5W。")
    answer = generated(
        "CKIN PET F02 的待机功耗为 5W。",
        "F02 待机功耗是 5W。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_ATTRIBUTE_VALUE_BINDING"


def test_guard_returns_claim_supporting_source_excerpt() -> None:
    evidence = result("前言介绍。该设备支持定时定量投喂。其他说明。")
    answer = generated(
        "该设备支持定时定量投喂。",
        "可以定时定量投喂。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is True
    assert dict(checked.source_excerpts)["S1"] == "该设备支持定时定量投喂。"


def test_guard_rejects_extra_source_that_supports_no_specific_claim() -> None:
    supporting = result("该设备支持定时定量投喂。", "S1")
    unrelated = result("该设备外包装为白色，适合家庭摆放。", "S2")
    answer = generated(
        "该设备支持定时定量投喂。",
        "可以定时定量投喂。",
        ["S1", "S2"],
    )

    checked = validate_generated_answer(answer, [supporting, unrelated])

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_SOURCE_MAPPING"
    assert checked.repair_detail is not None
    assert "S2" in checked.repair_detail


def test_guard_rejects_medical_claim_after_generation() -> None:
    evidence = result("饮水机提供循环活水。")
    answer = generated(
        "饮水机提供循环活水并能治疗疾病。",
        "该产品能够治疗疾病。",
        ["S1"],
    )

    checked = validate_generated_answer(answer, [evidence])

    assert checked.valid is False
    assert checked.reason == "FORBIDDEN_MEDICAL_CONTENT"


def test_guard_allows_non_actionable_veterinary_referral() -> None:
    answer = GeneralKnowledgeAnswer(
        title="日常护理",
        summary="本文只提供日常护理建议，不能替代兽医诊断或治疗。",
        sections=[
            AnswerSection(
                heading="安全提醒",
                content="如果宠物出现疼痛或出血，应停止操作并咨询执业兽医。",
            )
        ],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is True


def test_general_knowledge_guard_accepts_answer_without_document_sources() -> None:
    answer = GeneralKnowledgeAnswer(
        title="日常梳毛",
        summary="梳毛时动作保持轻柔，并观察宠物是否紧张。",
        sections=[AnswerSection(heading="步骤", content="先从容易接受的部位开始。")],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is True
    assert checked.cited_source_ids == ()


def test_general_knowledge_guard_rejects_fake_document_citation() -> None:
    answer = GeneralKnowledgeAnswer(
        title="日常梳毛",
        summary="梳毛时动作保持轻柔。[S1]",
        sections=[AnswerSection(heading="步骤", content="先从背部开始。")],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is False
    assert checked.reason == "UNEXPECTED_DOCUMENT_CITATION"


def test_general_knowledge_guard_rejects_household_disinfectant() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="保持猫砂盆干净。",
        sections=[AnswerSection(heading="清洗", content="定期用稀释消毒液清洗。")],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is False
    assert checked.reason == "UNSAFE_HOUSEHOLD_CHEMICAL"


def test_general_knowledge_guard_allows_explicit_chemical_warning() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="保持猫砂盆干净。",
        sections=[
            AnswerSection(
                heading="清洗",
                content="使用温水和无香温和清洁剂，并避免使用消毒液或精油。",
            )
        ],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is True


def test_general_knowledge_guard_allows_warning_after_chemical_name() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="高浓度 84 消毒液有刺激风险，不建议直接使用。",
        sections=[
            AnswerSection(
                heading="清洗",
                content="优先使用温水和无香温和清洁剂，并充分冲洗和晾干。",
            )
        ],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is True


def test_general_knowledge_guard_rejects_disinfectant_use_despite_later_rinse_warning() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="高浓度 84 消毒液有刺激风险，不建议直接使用。",
        sections=[
            AnswerSection(
                heading="清洗",
                content=("可考虑按说明书稀释后的 84 消毒液进行喷洒或擦拭，但必须充分冲洗并晾干。"),
            )
        ],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is False
    assert checked.reason == "UNSAFE_HOUSEHOLD_CHEMICAL"


def test_general_knowledge_salvage_drops_disinfectant_use_but_keeps_safe_alternative() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="不能用高浓度 84 消毒液清洁后立刻让猫使用。",
        sections=[
            AnswerSection(
                heading="安全替代",
                content=(
                    "优先使用温水和无香温和清洁剂，充分冲洗并晾干。"
                    "可考虑稀释后的 84 消毒液进行喷洒或擦拭，但必须充分冲洗。"
                ),
            )
        ],
        follow_up_questions=[],
    )

    salvaged = salvage_general_knowledge_answer(answer)

    assert salvaged is not None
    safe_answer, checked = salvaged
    assert checked.valid is True
    assert "不能用高浓度 84 消毒液" in safe_answer.summary
    assert "温水和无香温和清洁剂" in safe_answer.sections[0].content
    assert "喷洒或擦拭" not in safe_answer.sections[0].content


def test_general_knowledge_guard_does_not_treat_millet_sized_as_xiaomi_brand() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫粮颗粒大小",
        summary="小米粒大小描述的是颗粒尺寸，应结合成年猫的咀嚼情况选择。",
        sections=[AnswerSection(heading="观察", content="更换颗粒时逐步过渡。")],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer, ("小米",))

    assert checked.valid is True


def test_general_knowledge_guard_rejects_medical_benefit_claim() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="保持猫砂盆干净可以预防泌尿系统问题。",
        sections=[AnswerSection(heading="清理", content="每天清除排泄物。")],
        follow_up_questions=[],
    )

    checked = validate_general_knowledge_answer(answer)

    assert checked.valid is False
    assert checked.reason == "UNSUPPORTED_MEDICAL_BENEFIT"


def test_general_knowledge_salvage_removes_only_unsafe_benefit_sentence() -> None:
    answer = GeneralKnowledgeAnswer(
        title="猫砂盆清洁",
        summary="使用温水和无香温和清洁剂。这样可以预防泌尿系统问题。",
        sections=[
            AnswerSection(
                heading="安全清洁",
                content="充分冲洗并晾干。这样可以降低皮肤感染风险。",
            )
        ],
        follow_up_questions=[],
    )

    salvaged = salvage_general_knowledge_answer(answer)

    assert salvaged is not None
    safe_answer, checked = salvaged
    assert checked.valid is True
    assert "温水和无香温和清洁剂" in safe_answer.summary
    assert "预防" not in safe_answer.summary
    assert "充分冲洗并晾干" in safe_answer.sections[0].content
    assert "感染风险" not in safe_answer.sections[0].content


def test_general_knowledge_salvage_keeps_safe_clauses_when_every_sentence_has_medical_claim() -> (
    None
):
    answer = GeneralKnowledgeAnswer(
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

    salvaged = salvage_general_knowledge_answer(answer)

    assert salvaged is not None
    safe_answer, checked = salvaged
    assert checked.valid is True
    assert "定期开窗通风" in safe_answer.summary
    assert "及时清理排泄物" in safe_answer.sections[0].content
    assert "预防呼吸道疾病" not in safe_answer.summary
    assert "降低感染风险" not in safe_answer.sections[0].content
