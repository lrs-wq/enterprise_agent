import re
from dataclasses import dataclass
from decimal import Decimal

from company_agent.retrieval import SearchResult, lexical_score
from company_agent.schemas import GeneralKnowledgeAnswer, GeneratedAnswer


@dataclass(frozen=True)
class GuardResult:
    valid: bool
    reason: str
    cited_source_ids: tuple[str, ...] = ()
    source_excerpts: tuple[tuple[str, str], ...] = ()
    repair_detail: str | None = None
    failure_label: str | None = None
    failure_claim: str | None = None
    failure_source_ids: tuple[str, ...] = ()


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_CLAIM_SPLIT_PATTERN = re.compile(
    r"[。！？!?;；\n]+|(?:<br\s*/?>)+|(?:，|,)(?=\s*[^\d])",
    flags=re.IGNORECASE,
)
_EVIDENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?;；])\s*|\n+|(?=第\d+列\s*[:：])")
_MIN_CLAIM_OVERLAP = 0.08
_MIN_SOURCE_MAPPING_OVERLAP = 0.18

_UNIT_DEFINITIONS: dict[str, tuple[str, Decimal]] = {
    "万平方米": ("area_m2", Decimal("10000")),
    "万㎡": ("area_m2", Decimal("10000")),
    "万m²": ("area_m2", Decimal("10000")),
    "万m2": ("area_m2", Decimal("10000")),
    # Published OCR occasionally drops the square marker from area values.
    "万m": ("area_m2", Decimal("10000")),
    "万平": ("area_m2", Decimal("10000")),
    "平方米": ("area_m2", Decimal("1")),
    "平米": ("area_m2", Decimal("1")),
    "㎡": ("area_m2", Decimal("1")),
    "m²": ("area_m2", Decimal("1")),
    "m2": ("area_m2", Decimal("1")),
    "平": ("area_m2", Decimal("1")),
    "mah": ("battery_mah", Decimal("1")),
    "毫安时": ("battery_mah", Decimal("1")),
    "ghz": ("frequency_ghz", Decimal("1")),
    "毫升": ("volume_ml", Decimal("1")),
    "ml": ("volume_ml", Decimal("1")),
    "升": ("volume_ml", Decimal("1000")),
    "l": ("volume_ml", Decimal("1000")),
    "毫米": ("length_mm", Decimal("1")),
    "mm": ("length_mm", Decimal("1")),
    "厘米": ("length_mm", Decimal("10")),
    "cm": ("length_mm", Decimal("10")),
    "千米": ("length_mm", Decimal("1000000")),
    "km": ("length_mm", Decimal("1000000")),
    "米": ("length_mm", Decimal("1000")),
    "m": ("length_mm", Decimal("1000")),
    "毫克": ("mass_g", Decimal("0.001")),
    "mg": ("mass_g", Decimal("0.001")),
    "克": ("mass_g", Decimal("1")),
    "g": ("mass_g", Decimal("1")),
    "千克": ("mass_g", Decimal("1000")),
    "公斤": ("mass_g", Decimal("1000")),
    "kg": ("mass_g", Decimal("1000")),
    "斤": ("mass_g", Decimal("500")),
    "毫安": ("current_a", Decimal("0.001")),
    "ma": ("current_a", Decimal("0.001")),
    "安": ("current_a", Decimal("1")),
    "a": ("current_a", Decimal("1")),
    "伏": ("voltage_v", Decimal("1")),
    "v": ("voltage_v", Decimal("1")),
    "瓦": ("power_w", Decimal("1")),
    "w": ("power_w", Decimal("1")),
    "hz": ("frequency_hz", Decimal("1")),
    "db": ("noise_db", Decimal("1")),
    "%": ("percent", Decimal("1")),
    "°": ("degree", Decimal("1")),
    "度": ("degree", Decimal("1")),
    "秒": ("time_second", Decimal("1")),
    "s": ("time_second", Decimal("1")),
    "分钟": ("time_second", Decimal("60")),
    "min": ("time_second", Decimal("60")),
    "小时": ("time_second", Decimal("3600")),
    "h": ("time_second", Decimal("3600")),
    "天": ("time_second", Decimal("86400")),
    "年": ("year", Decimal("1")),
    "亩": ("area_mu", Decimal("1")),
    "元": ("currency_cny", Decimal("1")),
    "款": ("count_variant", Decimal("1")),
    "种": ("count_kind", Decimal("1")),
    "个": ("count_item", Decimal("1")),
    "台": ("count_device", Decimal("1")),
    "张": ("count_sheet", Decimal("1")),
    "项": ("count_item", Decimal("1")),
    "次": ("count_times", Decimal("1")),
    "档": ("count_level", Decimal("1")),
}
_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_UNIT_DEFINITIONS, key=len, reverse=True)
)
_MEASURED_NUMBER_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?P<operator>>=|<=|[≥≤><]|至少|不低于|不超过|至多|最高|最低|约为|大约|约)?\s*"
    rf"(?P<value>\d+(?:\.\d+)?)\s*"
    rf"(?P<post_operator>余|多|左右|上下|以上|以下)?\s*"
    rf"(?P<unit>{_UNIT_PATTERN})(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_BARE_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9.])(?P<value>\d+(?:\.\d+)?)(?![A-Za-z0-9.])")

_COUNT_UNIT_DEFINITIONS = {
    "中心": "count_center",
    "分公司": "count_branch",
    "品牌": "count_brand",
    "品类": "count_kind",
    "款": "count_variant",
    "种": "count_kind",
    "个": "count_item",
    "台": "count_device",
    "张": "count_sheet",
    "项": "count_item",
    "次": "count_times",
    "档": "count_level",
}
_COUNT_UNIT_PATTERN = "|".join(
    re.escape(unit) for unit in sorted(_COUNT_UNIT_DEFINITIONS, key=len, reverse=True)
)
_COUNT_NUMBER_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?P<value>\d+|[零一二三四五六七八九十两]+)\s*"
    rf"(?:大)?\s*(?P<unit>{_COUNT_UNIT_PATTERN})",
)

_MODEL_PATTERNS = (
    re.compile(
        r"(?:型号|款式|规格)\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]{0,19})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])([A-Za-z]{1,8}[-_ ]?\d{1,8}[A-Za-z0-9_-]*)"
        r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"((?:无线|插线|基础|视频|wifi|wi-fi|双仓|普通|标准|升级)款)",
        flags=re.IGNORECASE,
    ),
)

_ATTRIBUTE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"输入电压"), "input_voltage"),
    (re.compile(r"输出电压"), "output_voltage"),
    (re.compile(r"电压"), "voltage"),
    (re.compile(r"输入功率"), "input_power"),
    (re.compile(r"输出功率"), "output_power"),
    (re.compile(r"待机(?:功耗|功率)"), "standby_power"),
    (re.compile(r"(?:额定|工作)(?:功耗|功率)"), "rated_power"),
    (re.compile(r"功耗|功率"), "power"),
    (re.compile(r"电池容量|蓄电量"), "battery_capacity"),
    (re.compile(r"容量|容积|水箱|粮仓|储粮桶"), "capacity"),
    (re.compile(r"重量|净重|毛重"), "weight"),
    (re.compile(r"宽度|宽"), "width"),
    (re.compile(r"高度|高"), "height"),
    (re.compile(r"长度|长"), "length"),
    (re.compile(r"直径"), "diameter"),
    (re.compile(r"尺寸|规格"), "size"),
    (re.compile(r"续航|待机|时长|时间"), "duration"),
    (re.compile(r"角度|广角"), "angle"),
    (re.compile(r"工厂(?:整体|总)?面积|整体面积|总面积"), "total_area"),
    (re.compile(r"生产车间(?:面积)?"), "workshop_area"),
    (re.compile(r"办公室(?:面积)?"), "office_area"),
    (re.compile(r"(?:新建)?厂房(?:面积)?"), "factory_building_area"),
    (re.compile(r"(?:旧建筑|改造利用)(?:面积)?"), "renovated_area"),
    (re.compile(r"园区(?:土地)?面积|占地面积|占地"), "land_area"),
    (re.compile(r"面积|占地"), "area"),
    (re.compile(r"频率|网络"), "frequency"),
    (re.compile(r"噪音|分贝"), "noise"),
)

_NEGATIVE_POLARITY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"不支持|未支持|无法支持"), "支持"),
    (re.compile(r"不具备|未具备"), "具备"),
    (re.compile(r"未配备|不配备|没有配备"), "配备"),
    (re.compile(r"不兼容|无法兼容"), "兼容"),
    (re.compile(r"不包含|不含|未包含"), "包含"),
    (re.compile(r"不可使用|不能使用|无法使用"), "可使用"),
    (re.compile(r"没有"), "有"),
)
_POSITIVE_POLARITY_PATTERN = re.compile(
    r"支持|具备|配备|兼容|包含|可使用|可以|能够|提供|带有|拥有|内置|自带"
)

_QUALIFIER_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "availability": (
        ("optional", re.compile(r"可选|选配|需另购|可另配")),
        ("included", re.compile(r"标配|标准配置|随附|自带|内置")),
    ),
    "scope": (
        ("partial", re.compile(r"仅?部分(?:型号|款式|场景)?")),
        (
            "universal",
            re.compile(r"全部|所有|全系|(?:均|都)(?=支持|配备|具备|可)"),
        ),
    ),
}

_UNVERIFIED_SUBJECT_PATTERN = re.compile(
    r"(?:未登记|未知|未经审核|第三方).{0,8}(?:公司|品牌|产品)|"
    r"(?:公司|品牌|产品)[甲乙](?:的|值得|更好|推荐|$)"
)
_NON_FACTUAL_CLAIM_PATTERN = re.compile(
    r"^(?:(?:根据|依据)(?:知识库|已审核资料|资料|来源)(?:显示|记载|可知)?|"
    r"以下(?:为|是)?(?:主要)?(?:信息|内容|功能|参数)?|"
    r"(?:主要|具体)(?:如下|包括)?)$"
)


@dataclass(frozen=True)
class _Quantity:
    value: Decimal
    dimension: str
    condition: str
    start: int
    end: int


@dataclass(frozen=True)
class _Mention:
    value: str
    start: int
    end: int


def _normalize_unit(unit: str) -> str:
    return unit.lower().replace(" ", "")


def _unit_definition(unit: str) -> tuple[str, Decimal]:
    if unit == "G":
        return "frequency_ghz", Decimal("1")
    return _UNIT_DEFINITIONS[_normalize_unit(unit)]


def _chinese_count_value(value: str) -> Decimal:
    if value.isdigit():
        return Decimal(value)
    digits = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if "十" in value:
        left, right = value.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return Decimal((tens * 10) + ones)
    if len(value) == 1 and value in digits:
        return Decimal(digits[value])
    raise ValueError(f"不支持的中文计数：{value}")


def _normalize_condition(value: str | None, post_value: str | None = None) -> str:
    value = value or post_value
    if value in {">", "<"}:
        return value
    if value in {">=", "≥", "至少", "不低于", "最低", "以上"}:
        return "ge"
    if value in {"<=", "≤", "不超过", "至多", "最高", "以下"}:
        return "le"
    if value in {"约为", "大约", "约", "余", "多", "左右", "上下"}:
        return "approx"
    return "eq"


def _conditions_compatible(claim: str, evidence: str) -> bool:
    if claim == evidence:
        return True
    return evidence == "eq" and claim in {"ge", "le", "approx"}


def _extract_quantities(text: str) -> list[_Quantity]:
    quantities: list[_Quantity] = []
    occupied: list[tuple[int, int]] = []
    for match in _COUNT_NUMBER_PATTERN.finditer(text):
        quantities.append(
            _Quantity(
                value=_chinese_count_value(match.group("value")),
                dimension=_COUNT_UNIT_DEFINITIONS[match.group("unit")],
                condition="eq",
                start=match.start(),
                end=match.end(),
            )
        )
        occupied.append(match.span())
    for match in _MEASURED_NUMBER_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        dimension, multiplier = _unit_definition(match.group("unit"))
        quantities.append(
            _Quantity(
                value=Decimal(match.group("value")) * multiplier,
                dimension=dimension,
                condition=_normalize_condition(
                    match.group("operator"),
                    match.group("post_operator"),
                ),
                start=match.start(),
                end=match.end(),
            )
        )
        occupied.append(match.span())
    for match in _BARE_NUMBER_PATTERN.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        prefix = text[max(0, match.start() - 8) : match.start()]
        suffix = text[match.end() : match.end() + 3]
        if re.search(r"(?:^|[\n>:：])\s*$", prefix) and re.match(r"\s*[.)、]", suffix):
            continue
        quantities.append(
            _Quantity(
                value=Decimal(match.group("value")),
                dimension="bare_number",
                condition="eq",
                start=match.start(),
                end=match.end(),
            )
        )
    return sorted(quantities, key=lambda item: item.start)


def _model_mentions(text: str) -> list[_Mention]:
    mentions: list[_Mention] = []
    seen: set[tuple[int, int]] = set()
    for pattern in _MODEL_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span(1)
            if span in seen:
                continue
            seen.add(span)
            value = re.sub(r"[\s_-]+", "", match.group(1)).upper()
            mentions.append(_Mention(value=value, start=span[0], end=span[1]))
    return mentions


def _attribute_mentions(text: str) -> list[_Mention]:
    mentions: list[_Mention] = []
    for pattern, value in _ATTRIBUTE_PATTERNS:
        for match in pattern.finditer(text):
            mentions.append(_Mention(value=value, start=match.start(), end=match.end()))
    return mentions


def _sentence_span(text: str, position: int) -> tuple[int, int]:
    boundary = "。！？!?;；\n"
    start = max((text.rfind(char, 0, position) for char in boundary), default=-1) + 1
    ends = [text.find(char, position) for char in boundary]
    valid_ends = [end for end in ends if end >= 0]
    return start, min(valid_ends, default=len(text))


def _nearest_mention(text: str, quantity: _Quantity, mentions: list[_Mention]) -> str | None:
    sentence_start, sentence_end = _sentence_span(text, quantity.start)
    candidates = [
        mention
        for mention in mentions
        if mention.start >= sentence_start and mention.end <= sentence_end
    ]
    if not candidates:
        return None

    def distance(mention: _Mention) -> int:
        if mention.end <= quantity.start:
            return quantity.start - mention.end
        if mention.start >= quantity.end:
            return mention.start - quantity.end
        return 0

    return min(candidates, key=distance).value


def _attributes_compatible(claim_attribute: str, evidence_attribute: str) -> bool:
    if claim_attribute == evidence_attribute:
        return True
    power_attributes = {
        "power",
        "input_power",
        "output_power",
        "standby_power",
        "rated_power",
    }
    if claim_attribute in power_attributes and evidence_attribute in power_attributes:
        return claim_attribute == "power"
    generic_groups = (
        {"voltage", "input_voltage", "output_voltage"},
        {"size", "width", "height", "length", "diameter"},
        {
            "area",
            "total_area",
            "workshop_area",
            "office_area",
            "factory_building_area",
            "renovated_area",
            "land_area",
        },
    )
    return any(
        claim_attribute in group
        and evidence_attribute in group
        and (
            claim_attribute in {"voltage", "power", "size", "area"}
            or evidence_attribute in {"voltage", "power", "size", "area"}
        )
        for group in generic_groups
    )


def _quantity_support_reason(claim: str, evidence: str) -> str | None:
    claim_quantities = _extract_quantities(claim)
    if not claim_quantities:
        return None
    evidence_quantities = _extract_quantities(evidence)
    claim_models = _model_mentions(claim)
    evidence_models = _model_mentions(evidence)
    claim_attributes = _attribute_mentions(claim)
    evidence_attributes = _attribute_mentions(evidence)
    for claim_quantity in claim_quantities:
        matching_values = [
            item
            for item in evidence_quantities
            if item.dimension == claim_quantity.dimension and item.value == claim_quantity.value
        ]
        if not matching_values:
            return "UNSUPPORTED_MEASURED_VALUE"
        claim_model = _nearest_mention(claim, claim_quantity, claim_models)
        claim_attribute = _nearest_mention(claim, claim_quantity, claim_attributes)
        condition_mismatch = False
        model_mismatch = False
        attribute_mismatch = False
        for evidence_quantity in matching_values:
            if not _conditions_compatible(claim_quantity.condition, evidence_quantity.condition):
                condition_mismatch = True
                continue
            evidence_model = _nearest_mention(evidence, evidence_quantity, evidence_models)
            if claim_model and evidence_model and claim_model != evidence_model:
                model_mismatch = True
                continue
            evidence_attribute = _nearest_mention(evidence, evidence_quantity, evidence_attributes)
            if (
                claim_attribute
                and evidence_attribute
                and not _attributes_compatible(claim_attribute, evidence_attribute)
            ):
                attribute_mismatch = True
                continue
            return None
        if condition_mismatch:
            return "UNSUPPORTED_VALUE_CONDITION"
        if model_mismatch:
            return "UNSUPPORTED_SUBJECT_VALUE_BINDING"
        if attribute_mismatch:
            return "UNSUPPORTED_ATTRIBUTE_VALUE_BINDING"
        return "UNSUPPORTED_MEASURED_VALUE"
    return None


def _polarity(text: str) -> tuple[str, str] | None:
    for pattern, replacement in _NEGATIVE_POLARITY_PATTERNS:
        if pattern.search(text):
            return "negative", pattern.sub(replacement, text)
    if _POSITIVE_POLARITY_PATTERN.search(text):
        return "positive", text
    return None


def _qualifiers(text: str) -> tuple[dict[str, str], str]:
    values: dict[str, str] = {}
    normalized = text
    for group, patterns in _QUALIFIER_PATTERNS.items():
        for value, pattern in patterns:
            if pattern.search(text):
                values[group] = value
                normalized = pattern.sub("", normalized)
                break
    return values, normalized


def _models_compatible(claim: str, evidence: str) -> bool:
    claim_models = {mention.value for mention in _model_mentions(claim)}
    evidence_models = {mention.value for mention in _model_mentions(evidence)}
    return not claim_models or not evidence_models or bool(claim_models & evidence_models)


def _semantic_contradiction_reason(claim: str, evidence: str) -> str | None:
    claim_polarity = _polarity(claim)
    claim_qualifiers, qualifier_neutral_claim = _qualifiers(claim)
    evidence_segments = _evidence_segments(evidence)

    if claim_polarity:
        claim_value, polarity_neutral_claim = claim_polarity
        best: tuple[float, str] | None = None
        for segment in evidence_segments:
            if not _models_compatible(claim, segment):
                continue
            evidence_polarity = _polarity(segment)
            if not evidence_polarity:
                continue
            evidence_value, polarity_neutral_evidence = evidence_polarity
            score = lexical_score(polarity_neutral_claim, polarity_neutral_evidence)
            if best is None or score > best[0]:
                best = (score, evidence_value)
        if best and best[0] >= 0.45 and best[1] != claim_value:
            return "CONTRADICTED_CLAIM_POLARITY"

    for group, claim_value in claim_qualifiers.items():
        best_qualifier: tuple[float, str | None] | None = None
        for segment in evidence_segments:
            if not _models_compatible(claim, segment):
                continue
            evidence_qualifiers, neutral_evidence = _qualifiers(segment)
            score = lexical_score(qualifier_neutral_claim, neutral_evidence)
            if best_qualifier is None or score > best_qualifier[0]:
                best_qualifier = (score, evidence_qualifiers.get(group))
        if best_qualifier and best_qualifier[0] >= 0.45:
            qualifier_evidence_value = best_qualifier[1]
            if qualifier_evidence_value and qualifier_evidence_value != claim_value:
                return "CONTRADICTED_CLAIM_QUALIFIER"
            if qualifier_evidence_value is None and claim_value == "universal":
                return "UNSUPPORTED_CLAIM_QUALIFIER"
    return None


def _evidence_segments(text: str) -> list[str]:
    segments = [segment.strip() for segment in _EVIDENCE_SPLIT_PATTERN.split(text)]
    return [segment for segment in segments if segment]


def _atomic_claims(text: str) -> list[str]:
    clean = _strip_citations(text)
    claims = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", item).strip(" :：")
        for item in _CLAIM_SPLIT_PATTERN.split(clean)
    ]
    return [
        claim
        for claim in claims
        if len(re.sub(r"\W", "", claim)) >= 2 and not _NON_FACTUAL_CLAIM_PATTERN.fullmatch(claim)
    ]


def _metadata_claims(text: str) -> list[str]:
    return [
        claim
        for claim in _atomic_claims(text)
        if _extract_quantities(claim) or _polarity(claim) or _qualifiers(claim)[0]
    ]


def _best_supporting_segment(claim: str, evidence: str) -> tuple[float, str]:
    segments = _evidence_segments(evidence)
    if not segments:
        return 0.0, ""
    scored = [(lexical_score(claim, segment), segment) for segment in segments]
    return max(scored, key=lambda item: item[0])


def _excerpt_window(segment: str, claim: str, limit: int = 320) -> str:
    compact = " ".join(segment.split())
    if len(compact) <= limit:
        return compact
    tokens = sorted(
        {token for token in re.findall(r"[A-Za-z0-9_-]+|[一-鿿]{2,}", claim)},
        key=len,
        reverse=True,
    )
    position = next(
        (
            compact.lower().find(token.lower())
            for token in tokens
            if token.lower() in compact.lower()
        ),
        0,
    )
    start = max(0, position - (limit // 3))
    end = min(len(compact), start + limit)
    start = max(0, end - limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def _visible_citations(text: str) -> list[str]:
    return _CITATION_PATTERN.findall(text)


def _strip_citations(text: str) -> str:
    return _CITATION_PATTERN.sub("", text).strip()


def _forbidden_content_reason(answer_text: str, forbidden_terms: tuple[str, ...]) -> str | None:
    lowered_answer = answer_text.lower()
    for term in forbidden_terms:
        term_lower = term.lower()
        candidate = lowered_answer
        if term_lower == "小米":
            candidate = candidate.replace("小米粒", "")
        if term_lower in candidate:
            return "FORBIDDEN_THIRD_PARTY_CONTENT"
    if _UNVERIFIED_SUBJECT_PATTERN.search(answer_text):
        return "UNVERIFIED_THIRD_PARTY_CONTENT"
    if re.search(
        r"(?:能够|可以|用于|帮助|辅助|声称).{0,8}(?:诊断|确诊|治疗|治愈)|"
        r"(?:建议|应当|需要|自行).{0,8}(?:用药|服药|治疗|使用抗生素)|"
        r"(?:吃|服|喂|使用).{0,6}(?:药|抗生素|阿莫西林|头孢)|"
        r"(?:药物|用药|阿莫西林|头孢).{0,10}(?:剂量|频次|疗程|毫克|mg)",
        answer_text,
        flags=re.IGNORECASE,
    ):
        return "FORBIDDEN_MEDICAL_CONTENT"
    return None


def _suggests_unsafe_household_chemical(answer_text: str) -> bool:
    """Allow explicit warnings, but reject positive/ambiguous chemical-use advice."""

    substances = re.compile(r"消毒液|漂白剂|84\s*消毒|精油", flags=re.IGNORECASE)
    positive_use = re.compile(r"使用|直接用|可用|用于|喷洒|擦拭|浸泡|混合|清洗|清洁|涂抹|可考虑")
    prohibited_use = re.compile(
        r"不要|不应|不可|避免|切勿|禁止|不建议|无需|不得|不能|不宜|请勿|勿用|"
        r"未使用|没有使用|远离"
    )
    explicit_warning = re.compile(
        r"不要|不应|不可|避免|切勿|禁止|不建议|无需|不得|不能|不宜|请勿|勿用|"
        r"危险|有害|刺激|中毒|"
        r"必须充分冲洗|彻底冲洗|完全晾干|远离宠物",
        flags=re.IGNORECASE,
    )
    for match in substances.finditer(answer_text):
        clause_start = (
            max(answer_text.rfind(mark, 0, match.start()) for mark in "。！？!?;；，,\n") + 1
        )
        clause_ends = [answer_text.find(mark, match.end()) for mark in "。！？!?;；，,\n"]
        valid_clause_ends = [value for value in clause_ends if value >= 0]
        clause_end = min(valid_clause_ends, default=len(answer_text))
        clause = answer_text[clause_start:clause_end]
        if positive_use.search(clause) and not prohibited_use.search(clause):
            return True
        sentence_start = (
            max(answer_text.rfind(mark, 0, match.start()) for mark in "。！？!?;；\n") + 1
        )
        sentence_ends = [answer_text.find(mark, match.end()) for mark in "。！？!?;；\n"]
        valid_ends = [value for value in sentence_ends if value >= 0]
        sentence_end = min(valid_ends, default=len(answer_text))
        sentence = answer_text[sentence_start:sentence_end]
        if not explicit_warning.search(sentence):
            return True
    return False


def validate_public_output_text(
    text: str,
    forbidden_terms: tuple[str, ...] = (),
) -> GuardResult:
    """Apply output-boundary checks to non-model text exposed by the API."""

    reason = _forbidden_content_reason(text, forbidden_terms)
    return GuardResult(False, reason) if reason else GuardResult(True, "OUTPUT_TEXT_ALLOWED")


def validate_generated_answer(
    answer: GeneratedAnswer,
    results: list[SearchResult],
    forbidden_terms: tuple[str, ...] = (),
) -> GuardResult:
    """Validate the model's explicit per-block evidence mapping.

    The model supplies source IDs separately from prose. The caller may render those
    IDs only after this guard confirms that they exist and support the corresponding
    block; the guard never infers a source from lexical similarity.
    """

    source_lookup = {item.source.source_id: item for item in results}
    answer_text = "\n".join(
        [
            answer.title,
            answer.summary,
            *(value for section in answer.sections for value in (section.heading, section.content)),
            *(step.summary for step in answer.thinking_summary),
            *answer.follow_up_questions,
        ]
    )
    output_check = validate_public_output_text(answer_text, forbidden_terms)
    if not output_check.valid:
        return output_check

    if answer.status == "insufficient_evidence":
        if answer.sections or answer.summary_source_ids:
            return GuardResult(False, "INVALID_EVIDENCE_GAP_PAYLOAD")
        if _CITATION_PATTERN.search(answer_text):
            return GuardResult(False, "UNEXPECTED_INLINE_CITATION")
        return GuardResult(True, "MODEL_EVIDENCE_GAP")

    allowed = set(source_lookup)
    blocks = [
        (
            "summary",
            _metadata_claims(answer.title) + _atomic_claims(answer.summary),
            answer.summary_source_ids,
            True,
        ),
        *(
            (
                f"section:{index}",
                _metadata_claims(section.heading) + _atomic_claims(section.content),
                section.source_ids,
                True,
            )
            for index, section in enumerate(answer.sections)
        ),
    ]
    if not answer.sections or any(not source_ids for _, _, source_ids, _ in blocks):
        return GuardResult(False, "MISSING_BLOCK_SOURCE_IDS")

    cited = list(
        dict.fromkeys(source_id for _, _, source_ids, _ in blocks for source_id in source_ids)
    )
    if not set(cited).issubset(allowed):
        return GuardResult(False, "UNKNOWN_SOURCE_ID")
    blocks.extend(
        (
            f"follow_up:{index}",
            claims,
            cited,
            False,
        )
        for index, question in enumerate(answer.follow_up_questions)
        if (claims := _metadata_claims(question))
    )

    raw_blocks = [
        (answer.summary, answer.summary_source_ids),
        *((section.content, section.source_ids) for section in answer.sections),
    ]
    for raw_block, source_ids in raw_blocks:
        inline_ids = _visible_citations(raw_block)
        if inline_ids and not set(inline_ids).issubset(set(source_ids)):
            return GuardResult(False, "INLINE_SOURCE_MISMATCH")

    excerpt_candidates: dict[str, list[tuple[float, str]]] = {source_id: [] for source_id in cited}
    for label, claims, source_ids, require_all_sources in blocks:
        if not claims:
            return GuardResult(
                False,
                "EMPTY_SUBSTANTIVE_CLAIM",
                repair_detail=f"{label} 没有可核验的实质内容。",
            )
        block_supported_sources: set[str] = set()
        for claim in claims:
            claim_candidates: list[tuple[str, float, str]] = []
            failure_reasons: list[str] = []
            for source_id in source_ids:
                evidence = source_lookup[source_id].content
                score, segment = _best_supporting_segment(claim, evidence)
                if score < _MIN_CLAIM_OVERLAP:
                    continue
                quantity_reason = _quantity_support_reason(claim, evidence)
                if quantity_reason:
                    failure_reasons.append(quantity_reason)
                    continue
                semantic_reason = _semantic_contradiction_reason(claim, evidence)
                if semantic_reason:
                    failure_reasons.append(semantic_reason)
                    continue
                excerpt_reason = _forbidden_content_reason(segment, forbidden_terms)
                if excerpt_reason:
                    failure_reasons.append("FORBIDDEN_SOURCE_EXCERPT")
                    continue
                claim_candidates.append((source_id, score, segment))
            if not claim_candidates:
                priority = (
                    "CONTRADICTED_CLAIM_POLARITY",
                    "CONTRADICTED_CLAIM_QUALIFIER",
                    "UNSUPPORTED_CLAIM_QUALIFIER",
                    "UNSUPPORTED_VALUE_CONDITION",
                    "UNSUPPORTED_SUBJECT_VALUE_BINDING",
                    "UNSUPPORTED_ATTRIBUTE_VALUE_BINDING",
                    "UNSUPPORTED_MEASURED_VALUE",
                    "FORBIDDEN_SOURCE_EXCERPT",
                )
                reason = next(
                    (item for item in priority if item in failure_reasons),
                    "UNSUPPORTED_CLAIM",
                )
                return GuardResult(
                    False,
                    reason,
                    repair_detail=(
                        f"{label} 中的断言“{claim[:220]}”未被所列来源 "
                        f"{', '.join(source_ids)} 直接支持。"
                    ),
                    failure_label=label,
                    failure_claim=claim,
                )
            best_score = max(score for _source_id, score, _segment in claim_candidates)
            mapping_threshold = max(_MIN_SOURCE_MAPPING_OVERLAP, best_score * 0.6)
            for source_id, score, segment in claim_candidates:
                if score < mapping_threshold:
                    continue
                block_supported_sources.add(source_id)
                excerpt_candidates[source_id].append((score, _excerpt_window(segment, claim)))
        unmapped_sources = set(source_ids) - block_supported_sources
        if require_all_sources and unmapped_sources:
            return GuardResult(
                False,
                "UNSUPPORTED_SOURCE_MAPPING",
                repair_detail=(
                    f"{label} 列出的 {', '.join(sorted(unmapped_sources))} 没有直接支持"
                    "该回答块中的任何具体断言。"
                ),
                failure_label=label,
                failure_source_ids=tuple(sorted(unmapped_sources)),
            )

    excerpts: list[tuple[str, str]] = []
    for source_id in cited:
        candidates = sorted(excerpt_candidates[source_id], key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        for _score, excerpt in candidates:
            if excerpt not in selected:
                selected.append(excerpt)
            if len(" ".join(selected)) >= 240:
                break
        excerpts.append((source_id, " ".join(selected)[:320]))

    return GuardResult(
        valid=True,
        reason="SUPPORTED",
        cited_source_ids=tuple(cited),
        source_excerpts=tuple(excerpts),
    )


_SALVAGEABLE_GUARD_REASONS = {
    "CONTRADICTED_CLAIM_POLARITY",
    "CONTRADICTED_CLAIM_QUALIFIER",
    "UNSUPPORTED_CLAIM_QUALIFIER",
    "UNSUPPORTED_VALUE_CONDITION",
    "UNSUPPORTED_SUBJECT_VALUE_BINDING",
    "UNSUPPORTED_ATTRIBUTE_VALUE_BINDING",
    "UNSUPPORTED_MEASURED_VALUE",
    "UNSUPPORTED_CLAIM",
    "UNSUPPORTED_SOURCE_MAPPING",
}


def _remove_claim_from_text(text: str, claim: str) -> str:
    start = text.find(claim)
    if start < 0:
        return text
    before = text[:start]
    after = text[start + len(claim) :]
    left_separator = re.search(r"(?:[，,。！？!?；;][ \t]*)+$", before)
    right_separator = re.match(r"(?:[ \t]*[，,。！？!?；;])+", after)
    if left_separator and right_separator:
        marks = re.findall(
            r"[，,。！？!?；;]",
            left_separator.group() + right_separator.group(),
        )
        # Preserve the strongest sentence boundary when the removed claim sat
        # between two punctuation marks.
        separator = next(
            (mark for group in ("。！？!?", "；;", "，,") for mark in marks if mark in group),
            marks[0],
        )
        before = before[: left_separator.start()] + separator
        after = after[right_separator.end() :].lstrip(" \t")
    cleaned = before + after
    cleaned = re.sub(r"[，,]\s*[。！？!?；;]", "。", cleaned)
    cleaned = re.sub(r"([。！？!?；;])\s*\1+", r"\1", cleaned)
    return cleaned.strip().lstrip("，,。！？!?；;：:").rstrip("，,；;：:")


def _remove_failed_claim(answer: GeneratedAnswer, guard: GuardResult) -> bool:
    label = guard.failure_label
    claim = guard.failure_claim
    if not label or not claim:
        return False

    removed = False
    if label == "summary":
        if claim in _metadata_claims(answer.title):
            answer.title = "已核验信息"
            removed = True
        else:
            updated = _remove_claim_from_text(answer.summary, claim)
            removed = updated != answer.summary
            answer.summary = updated
    elif label.startswith("section:"):
        try:
            index = int(label.split(":", 1)[1])
        except ValueError:
            return False
        if index < 0 or index >= len(answer.sections):
            return False
        section = answer.sections[index]
        if claim in _metadata_claims(section.heading):
            section.heading = "相关信息"
            removed = True
        else:
            updated = _remove_claim_from_text(section.content, claim)
            removed = updated != section.content
            section.content = updated
            if removed and not _atomic_claims(section.content):
                answer.sections.pop(index)
    elif label.startswith("follow_up:"):
        try:
            index = int(label.split(":", 1)[1])
        except ValueError:
            return False
        if index < 0 or index >= len(answer.follow_up_questions):
            return False
        answer.follow_up_questions.pop(index)
        removed = True

    return removed


def _remove_inline_source_ids(text: str, source_ids: set[str]) -> str:
    for source_id in source_ids:
        text = re.sub(rf"\s*\[{re.escape(source_id)}\]", "", text)
    return text.strip()


def _remove_failed_source_mapping(answer: GeneratedAnswer, guard: GuardResult) -> bool:
    label = guard.failure_label
    failed_source_ids = set(guard.failure_source_ids)
    if not label or not failed_source_ids:
        return False

    if label == "summary":
        original = list(answer.summary_source_ids)
        answer.summary_source_ids = [
            source_id for source_id in original if source_id not in failed_source_ids
        ]
        answer.summary = _remove_inline_source_ids(answer.summary, failed_source_ids)
        return answer.summary_source_ids != original

    if label.startswith("section:"):
        try:
            index = int(label.split(":", 1)[1])
        except ValueError:
            return False
        if index < 0 or index >= len(answer.sections):
            return False
        section = answer.sections[index]
        original = list(section.source_ids)
        section.source_ids = [
            source_id for source_id in original if source_id not in failed_source_ids
        ]
        section.heading = _remove_inline_source_ids(section.heading, failed_source_ids)
        section.content = _remove_inline_source_ids(section.content, failed_source_ids)
        return section.source_ids != original

    return False


def _supported_summary_from_sections(
    answer: GeneratedAnswer,
    results: list[SearchResult],
    forbidden_terms: tuple[str, ...],
) -> tuple[str, list[str]] | None:
    for section in answer.sections:
        for claim in _atomic_claims(section.content):
            probe_section = section.model_copy(
                update={"heading": "相关信息", "content": claim},
                deep=True,
            )
            probe = answer.model_copy(
                update={
                    "title": "已核验信息",
                    "summary": claim,
                    "summary_source_ids": list(section.source_ids),
                    "sections": [probe_section],
                    "thinking_summary": [],
                    "follow_up_questions": [],
                },
                deep=True,
            )
            if validate_generated_answer(probe, results, forbidden_terms).valid:
                return claim, list(section.source_ids)
    return None


def salvage_supported_answer(
    answer: GeneratedAnswer,
    results: list[SearchResult],
    forbidden_terms: tuple[str, ...] = (),
    *,
    max_removed_claims: int = 48,
) -> tuple[GeneratedAnswer, GuardResult, int] | None:
    """Remove only failed atomic claims, then require the normal guard to pass in full."""

    candidate = answer.model_copy(deep=True)
    for removed_count in range(max_removed_claims + 1):
        guard = validate_generated_answer(candidate, results, forbidden_terms)
        if guard.valid:
            return candidate, guard, removed_count
        repaired = (
            _remove_failed_source_mapping(candidate, guard)
            if guard.reason == "UNSUPPORTED_SOURCE_MAPPING"
            else _remove_failed_claim(candidate, guard)
        )
        if (
            guard.reason not in _SALVAGEABLE_GUARD_REASONS
            or removed_count >= max_removed_claims
            or not repaired
            or not candidate.sections
        ):
            return None
        if not _atomic_claims(candidate.summary):
            replacement = _supported_summary_from_sections(candidate, results, forbidden_terms)
            if replacement is None:
                return None
            candidate.summary, candidate.summary_source_ids = replacement
    return None


def validate_general_knowledge_answer(
    answer: GeneralKnowledgeAnswer,
    forbidden_terms: tuple[str, ...] = (),
) -> GuardResult:
    """Validate model-internal pet-care output without inventing document evidence."""

    answer_text = "\n".join(
        [
            answer.title,
            answer.summary,
            *(value for section in answer.sections for value in (section.heading, section.content)),
            *answer.follow_up_questions,
        ]
    )
    output_check = validate_public_output_text(answer_text, forbidden_terms)
    if not output_check.valid:
        return output_check
    if _CITATION_PATTERN.search(answer_text):
        return GuardResult(False, "UNEXPECTED_DOCUMENT_CITATION")
    if not answer.sections:
        return GuardResult(False, "EMPTY_GENERAL_KNOWLEDGE_SECTIONS")
    if _GENERAL_PERSONALIZED_AMOUNT_PATTERN.search(answer_text):
        return GuardResult(False, "UNSAFE_PERSONALIZED_AMOUNT")
    if _suggests_unsafe_household_chemical(answer_text):
        return GuardResult(False, "UNSAFE_HOUSEHOLD_CHEMICAL")
    if _GENERAL_MEDICAL_BENEFIT_PATTERN.search(answer_text):
        return GuardResult(False, "UNSUPPORTED_MEDICAL_BENEFIT")
    return GuardResult(True, "MODEL_KNOWLEDGE")


_GENERAL_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?;；])\s*|\n+")
_GENERAL_CLAUSE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?;；，,])\s*|\n+")
_GENERAL_PERSONALIZED_AMOUNT_PATTERN = re.compile(
    r"(?:每天|每日|每顿).{0,12}\d+(?:\.\d+)?\s*(?:克|g|毫升|ml)",
    flags=re.IGNORECASE,
)
_GENERAL_MEDICAL_BENEFIT_PATTERN = re.compile(
    r"(?:预防|改善|缓解).{0,12}(?:疾病|感染|泌尿|肠胃|皮肤|呼吸|炎症|健康问题)|"
    r"降低.{0,12}(?:疾病|感染|泌尿|肠胃|皮肤|呼吸|炎症|健康问题)(?:风险)?",
    flags=re.IGNORECASE,
)


def _general_segment_is_safe(text: str, forbidden_terms: tuple[str, ...]) -> bool:
    if not text.strip():
        return False
    if not validate_public_output_text(text, forbidden_terms).valid:
        return False
    if _CITATION_PATTERN.search(text):
        return False
    if _GENERAL_PERSONALIZED_AMOUNT_PATTERN.search(text):
        return False
    if _suggests_unsafe_household_chemical(text):
        return False
    return _GENERAL_MEDICAL_BENEFIT_PATTERN.search(text) is None


def _safe_general_sentences(text: str, forbidden_terms: tuple[str, ...]) -> str:
    for pattern in (_GENERAL_SENTENCE_SPLIT_PATTERN, _GENERAL_CLAUSE_SPLIT_PATTERN):
        segments = [segment.strip() for segment in pattern.split(text) if segment.strip()]
        retained = "".join(
            segment for segment in segments if _general_segment_is_safe(segment, forbidden_terms)
        )
        if retained:
            # A removed clause may leave a dangling comma; make the retained
            # advice a complete sentence before validating the whole answer.
            return (
                retained.rstrip("，,；; ").rstrip() + "。"
                if retained.endswith(("，", ",", "；", ";"))
                else retained
            )
    return ""


def salvage_general_knowledge_answer(
    answer: GeneralKnowledgeAnswer,
    forbidden_terms: tuple[str, ...] = (),
) -> tuple[GeneralKnowledgeAnswer, GuardResult] | None:
    """Remove unsafe clauses after one failed repair, then rerun the full guard."""

    candidate = answer.model_copy(deep=True)
    if not _general_segment_is_safe(candidate.title, forbidden_terms):
        candidate.title = "日常养宠建议"
    candidate.summary = _safe_general_sentences(candidate.summary, forbidden_terms)

    safe_sections = []
    for section in candidate.sections:
        content = _safe_general_sentences(section.content, forbidden_terms)
        if not content:
            continue
        heading = section.heading
        if not _general_segment_is_safe(heading, forbidden_terms):
            heading = "日常建议"
        safe_sections.append(section.model_copy(update={"heading": heading, "content": content}))
    candidate.sections = safe_sections
    candidate.follow_up_questions = [
        question
        for question in candidate.follow_up_questions
        if _general_segment_is_safe(question, forbidden_terms)
    ]
    if not candidate.summary and candidate.sections:
        candidate.summary = candidate.sections[0].content[:1200]
    if not candidate.summary or not candidate.sections:
        return None

    guard = validate_general_knowledge_answer(candidate, forbidden_terms)
    if not guard.valid:
        return None
    return candidate, guard
