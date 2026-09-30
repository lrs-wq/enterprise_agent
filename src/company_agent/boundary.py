import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from company_agent.schemas import EvidenceStatus, ResponseStatus

BoundaryMode = Literal["retrieve", "general_model", "fixed"]
_IMPLICIT_PET_SOCIAL_PATTERN = re.compile(r"(?:同类|同伴).{0,10}(?:玩|相处|互动)")
_NON_PET_SOCIAL_OBJECT_PATTERN = re.compile(r"产品|商品|设备|机型|型号|玩具|游戏|公司|品牌")


@dataclass(frozen=True)
class BoundaryDecision:
    mode: BoundaryMode
    status: ResponseStatus
    evidence_status: EvidenceStatus
    reason_code: str
    policy_version: str
    intent: str
    domains: tuple[str, ...] = ()
    title: str = ""
    summary: str = ""
    refusal_reason: str | None = None
    follow_up_questions: tuple[str, ...] = ()
    partial_notice: str | None = None
    partial_kind: Literal["third_party", "medical"] | None = None
    verify_topic: bool = False


class AnswerBoundary:
    """Deterministic request gate that runs before retrieval or generation."""

    def __init__(self, policy_path: Path) -> None:
        with policy_path.open("r", encoding="utf-8") as handle:
            policy: dict[str, Any] = yaml.safe_load(handle)
        self.policy = policy
        self.version = str(policy["version"])
        self.templates: dict[str, dict[str, str]] = policy["templates"]
        self.blocked_third_party_aliases = tuple(
            str(item) for item in policy.get("blocked_third_party_aliases", [])
        )
        routing = policy["routing"]
        self.self_references = tuple(str(item).lower() for item in routing["self_references"])
        self.pet_subjects = tuple(str(item).lower() for item in routing["pet_subjects"])
        self.pet_care_intents = tuple(str(item).lower() for item in routing["pet_care_intents"])
        company = policy["company"]
        self.company_name = str(company["name"])
        aliases = list(company["aliases"])
        for brand in company["brands"]:
            aliases.extend(brand["aliases"])
        self.owned_aliases = tuple(str(item).lower() for item in aliases)
        self.owned_product_models = tuple(
            str(item).upper() for item in company.get("product_models", [])
        )

    @staticmethod
    def _contains(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _contains_term(text: str, terms: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in terms)

    @staticmethod
    def _third_party_mentions(text: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
        """Return named third parties without treating common compounds as brands."""

        lowered = text.lower()
        mentions: list[str] = []
        for alias in aliases:
            alias_lower = alias.lower()
            positions = [match.start() for match in re.finditer(re.escape(alias_lower), lowered)]
            if alias_lower == "小米":
                positions = [
                    position
                    for position in positions
                    if not lowered[position : position + len(alias_lower) + 1].startswith("小米粒")
                ]
            if positions:
                mentions.append(alias)
        return tuple(mentions)

    @staticmethod
    def _medical_scan_text(text: str) -> str:
        """Drop an explicitly negated medical request before scanning for active intent."""

        negated_medical_request = (
            r"(?:不需要|无需|不想|不是(?:想|要)?|并非(?:想|要)?).{0,8}?"
            r"(?:诊断|确诊|治疗|用药|药物|处方|吃药|打针|手术)"
            r"(?:建议|方案|意见)?"
            r"(?=\s*[，,;；。]?\s*(?:只|只是|而是|我只|我只是))"
        )
        return re.sub(negated_medical_request, "", text, flags=re.IGNORECASE)

    def needs_pet_subject_context(self, question: str) -> bool:
        """Detect a social-play follow-up whose pet subject was left unstated."""

        return (
            _IMPLICIT_PET_SOCIAL_PATTERN.search(question) is not None
            and not self.has_explicit_pet_subject(question)
            and _NON_PET_SOCIAL_OBJECT_PATTERN.search(question) is None
        )

    def has_explicit_pet_subject(self, question: str) -> bool:
        """Whether the current user question names a recognized pet subject."""

        return self._contains_term(question, self.pet_subjects)

    def _fixed(
        self,
        template: str,
        *,
        status: ResponseStatus,
        reason_code: str,
        intent: str = "other",
        follow_ups: tuple[str, ...] = (),
        summary: str | None = None,
        refusal_reason: str | None = None,
    ) -> BoundaryDecision:
        value = self.templates[template]
        return BoundaryDecision(
            mode="fixed",
            status=status,
            evidence_status="not_applicable",
            reason_code=reason_code,
            policy_version=self.version,
            intent=intent,
            title=value["title"],
            summary=summary or value["summary"],
            refusal_reason=refusal_reason or value.get("reason"),
            follow_up_questions=follow_ups,
        )

    def resolve_topic(
        self,
        decision: BoundaryDecision,
        kind: Literal[
            "company",
            "pet_product",
            "pet_care",
            "medical",
            "third_party",
            "pet_other",
            "non_pet",
            "uncertain",
        ],
    ) -> BoundaryDecision:
        if kind == "pet_care":
            if decision.mode == "general_model":
                return decision
            return BoundaryDecision(
                mode="general_model",
                status="general_knowledge",
                evidence_status="model_knowledge",
                reason_code="PET_CARE_GENERAL_MODEL_KNOWLEDGE",
                policy_version=self.version,
                intent="pet_care_general",
            )
        if kind == "company":
            if decision.mode == "retrieve" and decision.intent == "company_intro":
                return decision
            return BoundaryDecision(
                mode="retrieve",
                status="grounded",
                evidence_status="supported",
                reason_code="COMPANY_PUBLIC_ALLOWED",
                policy_version=self.version,
                intent="company_intro",
                domains=tuple(self.policy["allowed_domains"]["company"]),
            )
        if kind == "pet_product":
            if decision.mode == "retrieve" and decision.intent == "product_intro":
                return decision
            return BoundaryDecision(
                mode="retrieve",
                status="grounded",
                evidence_status="supported",
                reason_code="PET_PRODUCT_CATEGORY_ALLOWED",
                policy_version=self.version,
                intent="product_intro",
                domains=tuple(self.policy["allowed_domains"]["product"]),
            )
        if kind == "medical":
            return self._fixed(
                "medical",
                status="out_of_scope",
                reason_code="PET_MEDICAL_ADVICE",
                intent="pet_medical",
            )
        if kind == "third_party":
            label = "所提及的第三方品牌"
            return self._fixed(
                "other_company",
                status="out_of_scope",
                reason_code="THIRD_PARTY_BRAND",
                summary=self.templates["other_company"]["summary"].format(third_party=label),
                refusal_reason=self.templates["other_company"]["reason"].format(third_party=label),
            )
        if kind in {"pet_other", "non_pet"}:
            return self._fixed(
                "unrelated",
                status="out_of_scope",
                reason_code="UNRELATED_TOPIC",
            )
        if decision.intent == "pet_care_general":
            return self._fixed(
                "clarification_pet_subject",
                status="clarification_needed",
                reason_code="PET_SUBJECT_AMBIGUOUS",
                intent="pet_care_general",
                follow_ups=("请说明宠物种类或犬猫品种",),
            )
        return self._fixed(
            "clarification_topic",
            status="clarification_needed",
            reason_code="QUESTION_TOPIC_AMBIGUOUS",
            follow_ups=("请说明想了解公司产品，还是日常养宠问题",),
        )

    def classify(self, question: str) -> BoundaryDecision:
        text = " ".join(question.strip().split())
        lowered = text.lower()
        owned_brand_mentioned = any(alias in lowered for alias in self.owned_aliases)
        self_subject_mentioned = self._contains_term(text, self.self_references)
        medical_scan_text = self._medical_scan_text(text)

        emergency = (
            r"中毒|误食.*(?:毒|药|清洁剂)|呼吸困难|抽搐|昏迷|大出血|无法呼吸|休克|尿闭|吞食异物",
        )
        if self._contains(medical_scan_text, emergency):
            return self._fixed(
                "emergency",
                status="out_of_scope",
                reason_code="MEDICAL_EMERGENCY",
                intent="pet_medical",
            )

        medical = (
            r"诊断|确诊|什么病|治疗|治好|处方|用药|吃什么药|药量|剂量|打针|手术|消炎药|抗生素",
            r"阿莫西林|头孢|布洛芬|对乙酰氨基酚|(?:吃|服|喂).{0,10}(?:多少)?\s*毫克",
            r"呕吐|吐血|便血|尿血|腹泻|发烧|皮肤病|真菌|骨折|伤口|感染|不吃不喝",
            r"口吐白沫|大量流口水|流涎|站立不稳|精神萎靡|呼吸急促|排尿困难",
        )
        if self._contains(medical_scan_text, medical):
            separable = self._contains(text, (r"另外|同时|还有|顺便|并且|以及|[，,；;]",))
            independent_allowed_request = self._contains(
                text,
                (
                    r"介绍|说明|功能|参数|图片|照片|外观|怎么用|如何使用|"
                    r"怎么设置|配网|清洁|清洗|售后|故障|退款",
                ),
            )
            if (
                separable
                and independent_allowed_request
                and (owned_brand_mentioned or self_subject_mentioned)
            ):
                product_topic = owned_brand_mentioned or self._contains(
                    text,
                    (
                        r"产品|喂食器|饮水机|磨甲器|剃毛器|理毛器|烘干|"
                        r"猫砂盆|围栏|尿垫|玩具|配网|清洁|清洗",
                    ),
                )
                template = self.templates["mixed_medical"]
                return BoundaryDecision(
                    mode="retrieve",
                    status="partial_answer",
                    evidence_status="partial",
                    reason_code=(
                        "SELF_PRODUCT_WITH_MEDICAL_PARTIAL"
                        if product_topic
                        else "SELF_COMPANY_WITH_MEDICAL_PARTIAL"
                    ),
                    policy_version=self.version,
                    intent="product_intro" if product_topic else "company_intro",
                    domains=tuple(
                        self.policy["allowed_domains"]["product" if product_topic else "company"]
                    ),
                    refusal_reason=template["reason"],
                    partial_notice=template["summary"],
                    partial_kind="medical",
                    follow_up_questions=("如宠物症状持续或加重，请尽快联系兽医",),
                )
            return self._fixed(
                "medical",
                status="out_of_scope",
                reason_code="PET_MEDICAL_ADVICE",
                intent="pet_medical",
            )

        private_info = (
            r"内部成本|采购价|底价|毛利|员工(?:电话|手机|住址|身份证)|客户名单|未公开|商业秘密|后台密码",
            r"(?:负责人|员工|同事|个人|私人).{0,10}(?:电话|手机号|手机号码|住址|家庭地址)|家庭住址",
            r"忽略.{0,12}(?:规则|限制|系统)|我是(?:管理员|老板).{0,20}(?:内部|成本|密码)",
        )
        if self._contains(text, private_info):
            return self._fixed(
                "private_information",
                status="out_of_scope",
                reason_code="PRIVATE_OR_NONPUBLIC_INFORMATION",
            )

        exact_feeding = (
            r"(?:每天|一顿|每顿).{0,10}(?:多少|几)\s*(?:克|g|毫升|ml)",
            r"\d+(?:\.\d+)?\s*(?:kg|公斤|斤).{0,20}(?:吃|喂|喝).{0,10}(?:多少|几)",
        )
        if self._contains(text, exact_feeding):
            return self._fixed(
                "personalized_feeding",
                status="clarification_needed",
                reason_code="PERSONALIZED_FEEDING_AMOUNT",
                intent="pet_care_general",
                follow_ups=("补充食品名称、宠物年龄和体况",),
            )

        conversational = (
            r"^(你好|您好|嗨|hi|hello|在吗)[！!。.?？]?$",
            r"^(谢谢|感谢|好的|明白了|再见)[！!。.?？]?$",
            r"^(?:你能做什么|你是谁|你有什么功能|帮助)[！!。.?？]?$",
        )
        if self._contains(text, conversational):
            return self._fixed(
                "conversational",
                status="conversational",
                reason_code="CONVERSATIONAL",
                intent="conversation",
                follow_ups=("介绍一下公司", "CKIN PET 有哪些产品？"),
            )

        third_party_names = "|".join(re.escape(name) for name in self.blocked_third_party_aliases)
        other_company = (
            rf"(?:介绍|评价|推荐|分析).{{0,10}}(?:{third_party_names})",
            rf"(?:{third_party_names}).{{0,12}}(?:公司|品牌|产品|怎么样|哪个好|对比|比较)",
            r"(?:第三方|其他|别的|外部|竞品|友商)(?:公司|品牌|产品)",
        )
        comparison = self._contains(text, (r"对比|比较|哪个好|区别|优劣",))
        mentioned_third_parties = self._third_party_mentions(
            text,
            self.blocked_third_party_aliases,
        )
        named_third_party = bool(mentioned_third_parties)
        third_party = named_third_party or self._contains(text, other_company)
        third_party_label = "、".join(mentioned_third_parties) or "所提及的第三方品牌"
        third_party_summary = self.templates["other_company"]["summary"].format(
            third_party=third_party_label
        )
        third_party_reason = self.templates["other_company"]["reason"].format(
            third_party=third_party_label
        )
        if third_party and not owned_brand_mentioned:
            return self._fixed(
                "other_company",
                status="out_of_scope",
                reason_code="THIRD_PARTY_BRAND",
                intent="product_comparison" if comparison else "other",
                summary=third_party_summary,
                refusal_reason=third_party_reason,
            )

        partial_notice: str | None = None
        refusal_reason: str | None = None
        response_status: ResponseStatus = "grounded"
        evidence_status: EvidenceStatus = "supported"
        if third_party and owned_brand_mentioned:
            response_status = "partial_answer"
            evidence_status = "partial"
            partial_notice = self.templates["mixed_third_party"]["summary"].format(
                third_party=third_party_label
            )
            refusal_reason = self.templates["mixed_third_party"]["reason"].format(
                third_party=third_party_label
            )

        unknown_subject = (
            r"(?:未登记|未知|某个?|这家|那家).{0,12}(?:公司|品牌)",
            r"(?:介绍|了解|查询|评价|分析|推荐|说说|讲讲)(?:一下)?\s*"
            r"(?!公司|品牌|一下\s*(?:公司|品牌))"
            r"[^，。！？!?]{1,24}?(?:公司|品牌)",
        )
        if (
            not owned_brand_mentioned
            and not self_subject_mentioned
            and self._contains(text, unknown_subject)
        ):
            return self._fixed(
                "clarification_subject",
                status="clarification_needed",
                reason_code="SUBJECT_IDENTITY_AMBIGUOUS",
                follow_ups=("请提供准确的公司或品牌名称",),
            )

        ambiguous_product = (
            r"^(?:谢谢[，,]?\s*)?(?:(?:我家的|我的|家里的)\s*)?"
            r"(?:(?:它|这个|那个|这台|那台|这款|那款|该款|该产品|"
            r"你说的这款)(?:产品|设备|机器|喂食器|饮水机|猫砂盆)?|"
            r"(?:产品|设备|机器|喂食器|饮水机|猫砂盆))"
            r".{0,8}(?:怎么用|如何用|如何使用|怎么设置|怎么清洗|怎么清洁|怎么配网|说明书|"
            r"支持.{0,8}(?:控制|功能)|有哪些功能)",
            r"^[A-Za-z]{2,}[\s_-]*\d{2,}[A-Za-z0-9_-]*.{0,8}(?:怎么用|参数|功能|说明)",
        )
        incomplete_owned_model = self._contains(
            text,
            (
                r"(?:CKIN|Ckin)\s*(?:的\s*)?[A-Za-z](?![A-Za-z0-9])\s*.{0,8}"
                r"(?:配网|设置|参数|功能|怎么用)",
            ),
        )
        if incomplete_owned_model or (
            not owned_brand_mentioned
            and not self_subject_mentioned
            and self._contains(text, ambiguous_product)
        ):
            return self._fixed(
                "clarification_product",
                status="clarification_needed",
                reason_code="PRODUCT_IDENTITY_AMBIGUOUS",
                intent="product_intro",
                follow_ups=("请提供品牌、完整产品名或型号",),
            )

        unrelated = (
            r"写(?:一段|一?个)?\s*(?:python|java|javascript|代码|程序)|怎么编程|修复代码|数据库\s*sql|\bsql\b",
            r"天气|股票|基金|汇率|彩票|足球比分|电影推荐|推荐.{0,12}电影|旅游攻略|政治|总统|高考作文",
            r"(?:法国|英国|美国|日本|德国).{0,8}(?:首都|总统|人口)",
            r"(?:计算|算一下|帮我算).{0,20}\d+(?:\.\d+)?\s*"
            r"(?:加|减|乘以|乘|除以|[+\-*/×÷])\s*\d+(?:\.\d+)?",
        )
        if self._contains(text, unrelated):
            return self._fixed(
                "unrelated",
                status="out_of_scope",
                reason_code="UNRELATED_TOPIC",
            )

        if self.needs_pet_subject_context(text):
            return self._fixed(
                "clarification_pet_subject",
                status="clarification_needed",
                reason_code="PET_SUBJECT_AMBIGUOUS",
                intent="pet_care_general",
                follow_ups=("请说明宠物种类或犬猫品种",),
            )

        explicit_company_or_product = (
            r"公司|品牌|产品|业务|优势|资质|成立|地址|联系|售后|功能|参数|型号|图片|照片|外观|实拍",
        )
        has_company_topic = (
            self._contains(text, explicit_company_or_product)
            or owned_brand_mentioned
            or self_subject_mentioned
        )
        has_pet_subject = self._contains_term(text, self.pet_subjects)
        has_pet_care_intent = self._contains_term(text, self.pet_care_intents)
        pet_product_category = self._contains(
            text,
            (
                r"(?:宠物|猫咪?|狗狗?|犬|养宠|养猫|养狗)(?:用)?"
                r"(?:用品|商品|产品|设备|用具|器材|周边|装备)",
            ),
        )
        product_catalog_intent = self._contains(
            text,
            (
                r"有哪些|有什么|有啥|都卖|卖(?:什么|哪些)|主营|提供(?:什么|哪些)|"
                r"品类|种类|清单|目录|列出|介绍|推荐|展示",
            ),
        )
        pet_product_catalog = pet_product_category and product_catalog_intent
        has_company_topic = has_company_topic or pet_product_catalog

        company_identity_or_brand = self._contains(
            text,
            (
                r"(?:什么|哪家)公司(?:的)?(?:产品|品牌)?|属于.{0,10}(?:什么|哪家)?公司",
                r"品牌定位|同一个品牌|是你们的品牌|品牌关系",
                r"(?:概括|介绍).{0,12}公司.{0,12}品牌.{0,12}产品",
            ),
        )
        owned_product_model_mentioned = any(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(model)}(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
            for model in self.owned_product_models
        )
        explicit_owned_product = (
            self._contains(
                text,
                (
                    r"喂食器|饮水机|磨甲器|剃毛器|吹吸机|吹风机|烘干(?:机|箱)|猫砂盆|"
                    r"理毛器|狗厕所|围栏|宠物笼|尿垫|玩具|配网|出水模式|"
                    r"\b(?:feeder|water dispenser|dryer|groomer)\b",
                ),
            )
            or owned_product_model_mentioned
        )

        if (
            has_pet_subject
            and not owned_brand_mentioned
            and not self_subject_mentioned
            and has_pet_care_intent
            and not pet_product_catalog
        ):
            return BoundaryDecision(
                mode="general_model",
                status="general_knowledge",
                evidence_status="model_knowledge",
                reason_code="PET_CARE_GENERAL_MODEL_KNOWLEDGE",
                policy_version=self.version,
                intent="pet_care_general",
                verify_topic=True,
            )

        if not has_company_topic and not explicit_owned_product:
            return BoundaryDecision(
                mode="general_model",
                status="general_knowledge",
                evidence_status="model_knowledge",
                reason_code="PET_CARE_GENERAL_MODEL_KNOWLEDGE",
                policy_version=self.version,
                intent="pet_care_general",
                verify_topic=True,
            )

        product_topic = not company_identity_or_brand and (
            self._contains(
                text,
                (
                    r"产品|功能|参数|型号|喂食器|饮水机|磨甲器|剃毛器|吹吸机|"
                    r"理毛器|烘干|猫砂盆|围栏|尿垫|玩具|使用|配网|容量|功耗|噪声|"
                    r"材料牌号|兼容|出水模式|清洗|清洁",
                    r"\b(?:feeder|water dispenser|dryer|groomer)\b",
                ),
            )
            or explicit_owned_product
            or pet_product_catalog
            or (comparison and owned_brand_mentioned)
            or (
                partial_notice is not None
                and any(alias in lowered for alias in ("ckin pet", "ckinpet"))
            )
        )
        route = "product" if product_topic else "company"
        intent = (
            "product_comparison"
            if comparison
            else ("product_intro" if product_topic else "company_intro")
        )
        if partial_notice:
            reason_code = (
                "SELF_PRODUCT_ONLY_PARTIAL" if product_topic else "SELF_COMPANY_ONLY_PARTIAL"
            )
        elif pet_product_catalog and not owned_brand_mentioned and not self_subject_mentioned:
            reason_code = "PET_PRODUCT_CATEGORY_ALLOWED"
        else:
            reason_code = "OWNED_PRODUCT_ALLOWED" if product_topic else "COMPANY_PUBLIC_ALLOWED"
        return BoundaryDecision(
            mode="retrieve",
            status=response_status,
            evidence_status=evidence_status,
            reason_code=reason_code,
            policy_version=self.version,
            intent=intent,
            domains=tuple(self.policy["allowed_domains"][route]),
            refusal_reason=refusal_reason,
            partial_notice=partial_notice,
            partial_kind="third_party" if partial_notice else None,
            verify_topic=(
                not owned_brand_mentioned
                and not self_subject_mentioned
                and not owned_product_model_mentioned
            ),
        )
