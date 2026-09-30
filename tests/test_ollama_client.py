import json

import httpx

from company_agent.ollama_client import OllamaClient


def test_answer_profile_and_schema_skip_rebuilt_analysis() -> None:
    assert OllamaClient._answer_profile("宠沁和 CKIN PET 是同一个品牌吗？") == "brief"
    assert OllamaClient._answer_profile("请介绍 CKIN PET 的所有产品功能") == "standard"
    schema = OllamaClient._generated_schema()
    assert "thinking_summary" not in schema["properties"]
    assert "thinking_summary" not in schema["required"]


async def test_brief_fact_uses_short_budget_and_parses_without_analysis() -> None:
    payloads: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "status": "grounded",
                            "intent": "faq",
                            "title": "品牌关系",
                            "summary": "两个名称指向同一品牌。",
                            "summary_source_ids": ["S1"],
                            "sections": [
                                {
                                    "heading": "关系",
                                    "content": "两个名称指向同一品牌。",
                                    "source_ids": ["S1"],
                                }
                            ],
                        },
                        ensure_ascii=False,
                    )
                }
            },
        )

    client = OllamaClient("http://test", "model", "embedding", 4096)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001
    try:
        result = await client.answer(
            "宠沁和 CKIN PET 是同一个品牌吗？", "[S1] 已审核资料", [], ["S1"]
        )
    finally:
        await client.aclose()

    assert result.thinking_summary == []
    assert payloads[0]["options"]["num_predict"] == 900  # type: ignore[index]
    assert "thinking_summary" not in payloads[0]["format"]["properties"]  # type: ignore[index]


async def test_feeding_guidance_is_not_injected_into_bathing_answers() -> None:
    payloads: list[dict[str, object]] = []
    answer = {
        "title": "日常护理",
        "summary": "围绕当前问题回答。",
        "sections": [{"heading": "建议", "content": "只讨论当前主题。"}],
        "follow_up_questions": [],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"content": json.dumps(answer, ensure_ascii=False)}}
        )

    client = OllamaClient("http://test", "model", "embedding", 4096)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001
    try:
        bathing = await client.answer_general_knowledge("健康宠物日常洗澡注意什么？", [])
        await client.repair_general_knowledge(
            "健康宠物日常洗澡注意什么？", bathing, "UNSUPPORTED_MEDICAL_BENEFIT"
        )
        await client.answer_general_knowledge("喂食器日常怎么清洁？", [])
        feeding = await client.answer_general_knowledge("金毛每天应该喂多少？", [])
        await client.repair_general_knowledge(
            "金毛每天应该喂多少？", feeding, "UNSUPPORTED_FEEDING_QUANTITY"
        )
    finally:
        await client.aclose()

    for payload in payloads[:3]:
        prompt = json.dumps(payload["messages"], ensure_ascii=False)
        assert "食品包装" not in prompt
    for payload in payloads[3:]:
        prompt = json.dumps(payload["messages"], ensure_ascii=False)
        assert "食品包装" in prompt


async def test_hazardous_cleaning_question_gets_direct_no_and_no_usage_steps_guidance() -> None:
    payloads: list[dict[str, object]] = []
    answer = {
        "title": "猫砂盆清洁",
        "summary": "不能直接使用高浓度消毒液。",
        "sections": [{"heading": "安全替代", "content": "使用温水和无香温和清洁剂。"}],
        "follow_up_questions": [],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"content": json.dumps(answer, ensure_ascii=False)}}
        )

    client = OllamaClient("http://test", "model", "embedding", 4096)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))  # noqa: SLF001
    try:
        generated = await client.answer_general_knowledge(
            "能不能直接用高浓度 84 消毒液清洁猫砂盆后马上让猫使用？", []
        )
        await client.repair_general_knowledge(
            "能不能直接用高浓度 84 消毒液清洁猫砂盆后马上让猫使用？",
            generated,
            "UNSAFE_HOUSEHOLD_CHEMICAL",
        )
    finally:
        await client.aclose()

    first_prompt = json.dumps(payloads[0]["messages"], ensure_ascii=False)
    repair_prompt = json.dumps(payloads[1]["messages"], ensure_ascii=False)
    assert "请直接回答不能" in first_prompt
    assert "不给该化学品的稀释、喷洒、擦拭或浸泡步骤" in first_prompt
    assert "不得给出该化学品的使用步骤" in repair_prompt
