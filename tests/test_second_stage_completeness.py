from pathlib import Path

import pytest

from company_agent.config import Settings
from company_agent.database import KnowledgeRepository
from company_agent.schemas import ChatMessage, ChatRequest
from company_agent.services import AgentService


class _UnusedModel:
    pass


_HISTORIES = {
    "none": [],
    "water": [
        ChatMessage(role="user", content="CKIN PET F02 饮水机的容量是多少？"),
        ChatMessage(role="assistant", content="当前讨论的是 CKIN PET F02 饮水机。"),
    ],
    "feeder": [
        ChatMessage(role="user", content="请介绍 CKIN PET F02 视频款喂食器的功能。"),
        ChatMessage(role="assistant", content="当前讨论的是 CKIN PET F02 视频款喂食器。"),
    ],
    "third_party": [
        ChatMessage(role="user", content="请介绍小佩喂食器。"),
        ChatMessage(role="assistant", content="小佩是本公司产品。"),
    ],
    "multiple": [
        ChatMessage(role="user", content="请介绍 CKIN PET F01、F02 和 X01。"),
        ChatMessage(role="assistant", content="已介绍三个型号。"),
    ],
    "golden": [
        ChatMessage(role="user", content="金毛喜欢吃什么"),
        ChatMessage(role="assistant", content="金毛日常饮食应以营养均衡为原则。"),
    ],
}


_CASES = [
    ("它的容量是多少？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "饮水机"), ()),
    ("那容量呢？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "容量"), ()),
    ("只比较容量", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "容量"), ()),
    ("WIFI款呢？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "WIFI款", "容量"), ()),
    ("视频款呢？", "feeder", "OWNED_PRODUCT_ALLOWED", ("F02", "视频款", "功能"), ()),
    ("它怎么配网？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "配网"), ()),
    ("那怎么清洗？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "清洗"), ()),
    ("继续说功能", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "功能"), ()),
    ("还有图片吗？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "图片"), ()),
    ("参数呢？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "参数"), ()),
    ("X01呢？", "feeder", "OWNED_PRODUCT_ALLOWED", ("X01", "喂食器"), ("F02",)),
    ("换成F01呢？", "feeder", "OWNED_PRODUCT_ALLOWED", ("F01", "喂食器"), ("F02",)),
    ("它和小佩哪个好？", "feeder", "SELF_PRODUCT_ONLY_PARTIAL", ("F02", "小佩"), ()),
    ("小佩呢？", "water", "THIRD_PARTY_BRAND", ("小佩",), ("F02",)),
    (
        "猫咪平时怎么梳毛？",
        "water",
        "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
        ("猫咪",),
        ("F02",),
    ),
    ("猫咪呕吐该吃什么药？", "water", "PET_MEDICAL_ADVICE", ("呕吐",), ("F02",)),
    ("你好", "water", "CONVERSATIONAL", ("你好",), ("F02",)),
    ("今天天气怎么样？", "water", "UNRELATED_TOPIC", ("天气",), ("F02",)),
    ("它怎么清洁？", "multiple", "PRODUCT_IDENTITY_AMBIGUOUS", ("它",), ()),
    ("它怎么设置？", "third_party", "THIRD_PARTY_BRAND", ("小佩",), ()),
    ("CKIN PET X01 怎么用？", "water", "OWNED_PRODUCT_ALLOWED", ("X01",), ("F02",)),
    ("这个怎么设置？", "none", "PRODUCT_IDENTITY_AMBIGUOUS", ("这个",), ()),
    (
        "请介绍 CKIN PET F02 饮水机功能，另外猫咪呕吐了怎么办？",
        "none",
        "SELF_PRODUCT_WITH_MEDICAL_PARTIAL",
        ("F02", "呕吐"),
        (),
    ),
    (
        "介绍 CKIN PET，另外猫咪中毒抽搐怎么办？",
        "none",
        "MEDICAL_EMERGENCY",
        ("中毒",),
        (),
    ),
    (
        "用你们 F02 饮水机能治疗泌尿疾病吗？",
        "none",
        "PET_MEDICAL_ADVICE",
        ("F02", "治疗"),
        (),
    ),
    (
        "请介绍你们公司业务，另外猫咪呕吐了怎么办？",
        "none",
        "SELF_COMPANY_WITH_MEDICAL_PARTIAL",
        ("公司", "呕吐"),
        (),
    ),
    (
        "小佩和 CKIN PET F02 喂食器容量哪个好？",
        "none",
        "SELF_PRODUCT_ONLY_PARTIAL",
        ("F02", "容量"),
        (),
    ),
    ("它的图片呢？", "water", "OWNED_PRODUCT_ALLOWED", ("F02", "图片"), ()),
    ("那喂食器呢？", "water", "OWNED_PRODUCT_ALLOWED", ("喂食器",), ("F02",)),
    ("谢谢", "water", "CONVERSATIONAL", ("谢谢",), ("F02",)),
    (
        "它喜欢干什么",
        "golden",
        "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
        ("金毛", "喜欢干什么"),
        (),
    ),
]


@pytest.mark.parametrize(
    ("question", "history_key", "reason_code", "required", "forbidden"),
    _CASES,
)
def test_multiturn_and_mixed_request_matrix(
    tmp_path: Path,
    question: str,
    history_key: str,
    reason_code: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    assert len(_CASES) == 31
    settings = Settings(data_dir=tmp_path / "data")
    service = AgentService(
        settings,
        KnowledgeRepository(settings.database_path),
        _UnusedModel(),  # type: ignore[arg-type]
    )
    request = ChatRequest(question=question, history=_HISTORIES[history_key])

    resolved = service.resolve_question(request)
    decision = service.boundary.classify(resolved)

    assert decision.reason_code == reason_code
    assert all(value in resolved for value in required)
    assert all(value not in resolved for value in forbidden)
