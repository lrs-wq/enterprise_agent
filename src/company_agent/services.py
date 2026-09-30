import asyncio
import hashlib
import json
import logging
import re
import shutil
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from company_agent.answer_cache import PublicAnswerCache
from company_agent.answer_guard import (
    GuardResult,
    salvage_general_knowledge_answer,
    salvage_supported_answer,
    validate_general_knowledge_answer,
    validate_generated_answer,
    validate_public_output_text,
)
from company_agent.boundary import AnswerBoundary, BoundaryDecision
from company_agent.config import Settings
from company_agent.database import (
    PUBLIC_OWNER_COMPANY_ID,
    KnowledgeRepository,
    pack_embedding,
    utc_now,
)
from company_agent.document_parser import SUPPORTED_EXTENSIONS, ParsedChunk
from company_agent.isolated_tasks import (
    extract_document_media_isolated,
    parse_document_isolated,
)
from company_agent.ollama_client import (
    PROMPT_VERSION,
    OllamaClient,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from company_agent.product_identity import (
    ProductFacets,
    extract_product_facets,
    extract_product_identities,
    product_facets_match,
)
from company_agent.retrieval import SearchResult, build_context, lexical_score, retrieve
from company_agent.schemas import (
    AnswerPayload,
    AnswerSection,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DecisionView,
    FallbackKind,
    GeneralKnowledgeAnswer,
    GeneratedAnswer,
    GeneratedSection,
    ResponseImage,
    SourceRef,
    ThinkingStep,
)

logger = logging.getLogger(__name__)


class ModelStageBudgetExceeded(TimeoutError):
    def __init__(self, stage: str) -> None:
        super().__init__(f"模型{stage}阶段超出本次请求的剩余时限")
        self.stage = stage


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    normalized = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", name, flags=re.UNICODE)
    return normalized[:180] or "document"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


_INLINE_CITATION_PATTERN = re.compile(r"\[(?:S\d+)\]")
_IMAGE_REQUEST_PATTERN = re.compile(
    r"图片|照片|产品图|实拍|外观|长什么样|(?:展示|看看|看一下|看下).{0,8}(?:图|照片|外观)"
)
_TEXT_DETAIL_REQUEST_PATTERN = re.compile(
    r"介绍|说明|功能|参数|容量|尺寸|重量|怎么|如何|设置|配网|清洁|清洗|区别|比较"
)
_DEMONSTRATIVE_PATTERN = re.compile(r"它|这款|这个|这台|该款|该产品|你说的这款产品|刚才|上面")
_CONTEXT_REFERENCE_PATTERN = re.compile(r"你说的这款产品|这款产品|该产品|这台机器|它")
_PERSONAL_OR_VOLATILE_QUESTION_PATTERN = re.compile(
    r"我|我的|我们|本人|订单|手机号|电话|邮箱|住址|现在|今天|目前|最新|库存|价格"
)
_NON_PET_CONTEXT_BREAK_PATTERN = re.compile(r"养花|种花|花卉|植物|种植")
_FOLLOWUP_PATTERN = re.compile(
    r"^(?:那|那么|还有|另外|顺便|继续|再|也|只|仅|换成|如果是|至于)|"
    r"(?:呢|怎么样|如何)[？?]?$"
)
_PRODUCT_TYPE_PATTERN = re.compile(
    r"饮水机|喂食器|磨甲器|剃毛器|吹吸机|吹风机|烘干(?:机|箱)|"
    r"猫砂盆|狗厕所|围栏|宠物笼|尿垫|玩具"
)
_PRODUCT_VARIANT_PATTERN = re.compile(
    r"WIFI\s*款|Wi-Fi\s*款|视频款|无线款|插线款|基础款|双仓",
    flags=re.IGNORECASE,
)
_GENERIC_PET_SUBJECTS = frozenset(
    {
        "宠物",
        "猫",
        "猫咪",
        "小猫",
        "幼猫",
        "成猫",
        "老年猫",
        "狗",
        "狗狗",
        "小狗",
        "幼犬",
        "成犬",
        "成年犬",
        "老年犬",
    }
)
_CAPACITY_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>毫升|ml|升|l)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_EXTRACTIVE_SEGMENT_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?;；])\s*|\n+|\s{2,}")
_EXTRACTIVE_FALLBACK_REASONS = {
    "FORBIDDEN_MEDICAL_CONTENT",
    "FORBIDDEN_THIRD_PARTY_CONTENT",
    "INVALID_EVIDENCE_GAP_PAYLOAD",
    "MODEL_EVIDENCE_GAP",
    "MODEL_OUTPUT_INVALID",
    "UNSUPPORTED_CLAIM",
    "UNSUPPORTED_MEASURED_VALUE",
    "UNSUPPORTED_SOURCE_MAPPING",
    "UNVERIFIED_THIRD_PARTY_CONTENT",
}
_EXTRACTIVE_SKIP_PATTERN = re.compile(
    r"资料未写|未提及|未证实|全部参数|所有版本|到底",
    flags=re.IGNORECASE,
)
_DIRECT_EVIDENCE_REQUIREMENTS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (re.compile(r"售后|保修|退换|客服"), re.compile(r"售后|保修|退换|客服")),
    (
        re.compile(r"待机.{0,8}(?:功耗|多少瓦)|待机功耗"),
        re.compile(
            r"待机(?:状态)?(?:下)?\s*(?:功耗|功率)?\s*[:：为]?\s*"
            r"\d+(?:\.\d+)?\s*[W瓦]|(?:功耗|功率).{0,8}待机",
            re.IGNORECASE,
        ),
    ),
    (
        re.compile(r"(?:工作)?(?:噪声|噪音).{0,8}(?:精确|多少分贝)|精确.{0,8}分贝"),
        re.compile(r"\d+(?:\.\d+)?\s*(?:分贝|db)", re.IGNORECASE),
    ),
    (
        re.compile(r"功耗|多少瓦"),
        re.compile(r"功耗|功率|\d+(?:\.\d+)?\s*[W瓦]", re.IGNORECASE),
    ),
    (re.compile(r"噪声|噪音|分贝"), re.compile(r"噪声|噪音|分贝|db", re.IGNORECASE)),
    (re.compile(r"材料牌号"), re.compile(r"材料牌号|材质[:：]?\s*[A-Za-z0-9]")),
    (
        re.compile(r"兼容.*(?:iOS|Android)", re.IGNORECASE),
        re.compile(r"iOS.*Android", re.IGNORECASE),
    ),
    (re.compile(r"容量"), re.compile(r"容量|容积|\d\s*(?:L|升|毫升|ml)", re.IGNORECASE)),
    (re.compile(r"出水模式"), re.compile(r"出水模式|持续出水|感应出水|定时出水")),
    (re.compile(r"配网|绑定"), re.compile(r"配网|绑定")),
    (re.compile(r"清洁|清洗"), re.compile(r"清洁|清洗|擦拭|冲洗")),
)
_STRICT_EVIDENCE_REQUIREMENTS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (
        re.compile(r"售后流程"),
        re.compile(r"售后流程|售后.{0,12}(?:客服|退换|保修|处理步骤)"),
    ),
    (
        re.compile(r"20\d{2}\s*年.{0,16}(?:营收|净利润)|(?:营收|净利润).{0,16}20\d{2}\s*年"),
        re.compile(r"(?:营收|营业收入).{0,80}净利润|净利润.{0,80}(?:营收|营业收入)"),
    ),
    (
        re.compile(r"(?:现在|当前|实时).{0,12}库存|库存"),
        re.compile(r"库存(?:量|数量)?\s*[:：为]?\s*\d+|现货\s*[:：为]?\s*\d+"),
    ),
    (
        re.compile(r"(?:今天|当前|实时).{0,12}(?:新增)?订单|订单量"),
        re.compile(r"(?:今天|当日|当前|实时).{0,12}(?:新增)?订单.{0,8}\d+"),
    ),
    (
        re.compile(r"国际认证"),
        re.compile(r"(?:获得|通过|具备).{0,16}(?:国际认证|CE|FCC|RoHS)", re.IGNORECASE),
    ),
    (
        re.compile(r"保修.{0,6}(?:几年|年限|多久)"),
        re.compile(r"保修(?:期|年限)?\s*[:：为]?\s*\d+(?:\.\d+)?\s*年"),
    ),
    _DIRECT_EVIDENCE_REQUIREMENTS[1],
    _DIRECT_EVIDENCE_REQUIREMENTS[2],
    (
        re.compile(r"所有塑料部件.{0,8}材料牌号"),
        re.compile(r"所有塑料部件.{0,16}材料牌号|材料牌号.{0,16}所有塑料部件"),
    ),
    (
        re.compile(r"兼容所有版本.{0,12}(?:iOS|Android)", re.IGNORECASE),
        re.compile(
            r"(?:所有版本|全版本).{0,16}iOS.{0,16}Android|"
            r"iOS.{0,16}Android.{0,16}(?:所有版本|全版本)",
            re.IGNORECASE,
        ),
    ),
    (
        re.compile(
            r"(?:今天|当前|实时).{0,12}(?:售价|价格)|(?:售价|价格).{0,12}(?:今天|当前|实时)"
        ),
        re.compile(r"(?:当前|今日|实时).{0,8}(?:售价|价格)\s*[:：为]?\s*[¥￥]?\d+"),
    ),
    (
        re.compile(r"电源适配器.{0,10}输出电流"),
        re.compile(r"输出电流\s*[:：为]?\s*\d+(?:\.\d+)?\s*(?:mA|A|毫安|安)", re.IGNORECASE),
    ),
)
_BROAD_BUSINESS_PATTERN = re.compile(
    r"(?:公司|企业|工厂).{0,8}(?:主要做什么|主营(?:什么)?|从事什么|经营什么)(?:业务)?|"
    r"主营业务|主要业务|业务范围"
)
_EXPLICIT_BUSINESS_DETAIL_PATTERN = re.compile(
    r"面积|规模|占地|厂房|销售额|销量|比例|百分比|专利|知识产权|"
    r"多少|几个|几项|成立时间|哪一年"
)
_MEDIA_CODE_PATTERN = re.compile(r"(?<![A-Z0-9])([A-Z]{1,6}[- ]?\d{2,6})(?![A-Z0-9])")
_MEDIA_KEYWORDS = (
    "饮水机",
    "喂食器",
    "猫砂盆",
    "猫砂",
    "狗厕所",
    "围栏",
    "宠物笼",
    "尿垫",
    "玩具",
    "磨甲器",
    "剃毛器",
    "吹吸机",
    "吹风机",
    "烘干箱",
    "无线",
    "插线",
    "基础款",
    "wifi",
    "视频款",
)
_MEDIA_CATEGORY_KEYWORDS = (
    "饮水机",
    "喂食器",
    "猫砂盆",
    "猫砂",
    "狗厕所",
    "围栏",
    "宠物笼",
    "尿垫",
    "玩具",
    "磨甲器",
    "剃毛器",
    "吹吸机",
    "吹风机",
    "烘干箱",
)
_TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("图片", _IMAGE_REQUEST_PATTERN),
    ("容量", re.compile(r"容量|容积|多少升|多少毫升", re.I)),
    ("配网设置", re.compile(r"配网|联网|绑定|怎么设置|如何设置", re.I)),
    ("清洁方法", re.compile(r"清洁|清洗|拆洗", re.I)),
    ("功能", re.compile(r"功能|能做什么|支持什么", re.I)),
    ("参数", re.compile(r"参数|尺寸|重量|功耗|噪声|噪音", re.I)),
    ("适用场景", re.compile(r"适用|场景|适合", re.I)),
)
_REQUEST_ASPECT_PATTERN = re.compile(
    r"容量|功能|参数|图片|照片|外观|配网|设置|清洁|清洗|尺寸|重量|功耗|"
    r"噪声|噪音|适用场景|售后|使用方法"
)


@dataclass(frozen=True)
class ConversationState:
    subject: str | None = None
    topic: str = ""
    brand: str = ""
    model: str = ""
    category: str = ""
    variant: str = ""
    task: str = ""
    ambiguous: bool = False
    authorized: bool = True


def render_validated_citations(text: str, source_ids: list[str]) -> str:
    """Render only source IDs that already passed the per-block evidence guard."""

    clean = _INLINE_CITATION_PATTERN.sub("", text).strip()
    if not source_ids:
        return clean
    markers = "".join(f"[{source_id}]" for source_id in source_ids)
    return f"{clean} {markers}"


class KnowledgeService:
    def __init__(
        self,
        settings: Settings,
        repository: KnowledgeRepository,
        ollama: OllamaClient,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.ollama = ollama

    @staticmethod
    def _normalize_effective_time(value: object) -> str | None:
        if value is None:
            return None
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("生效时间必须包含时区")
        return parsed.astimezone(UTC).isoformat()

    def _normalize_document_values(self, values: dict[str, object]) -> dict[str, object]:
        normalized = dict(values)
        for field in ("effective_from", "effective_to"):
            if field in normalized:
                normalized[field] = self._normalize_effective_time(normalized[field])
        return normalized

    @staticmethod
    def _validate_effective_window(document: dict[str, object], values: dict[str, object]) -> None:
        start_raw = values.get("effective_from", document.get("effective_from"))
        end_raw = values.get("effective_to", document.get("effective_to"))
        if start_raw is None or end_raw is None:
            return
        start = datetime.fromisoformat(str(start_raw))
        end = datetime.fromisoformat(str(end_raw))
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("生效时间必须包含时区")
        if start > end:
            raise ValueError("生效开始时间不能晚于结束时间")

    def _copy_media_snapshot(
        self,
        document_id: str,
        version: int,
        assets: list[dict[str, object]],
    ) -> dict[str, str]:
        destination_dir = self.settings.published_media_dir / document_id / str(version)
        paths: dict[str, str] = {}
        for asset in assets:
            source = Path(str(asset["stored_path"])).resolve()
            enabled = bool(asset.get("enabled", False))
            if not source.is_file():
                if enabled:
                    raise ValueError(f"待发布图片文件不存在：{asset['filename']}")
                continue
            if not source.is_relative_to(
                self.settings.media_dir.resolve()
            ) and not source.is_relative_to(self.settings.published_media_dir.resolve()):
                raise ValueError("图片文件不在受管目录内")
            destination_dir.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(f"{asset['id']}-{asset['filename']}")
            destination = destination_dir / filename
            if destination.exists():
                if file_sha256(destination) != file_sha256(source):
                    raise ValueError("已发布版本的图片文件不可覆盖")
            else:
                shutil.copy2(source, destination)
            paths[str(asset["id"])] = str(destination)
        return paths

    def _materialize_active_media_snapshot(self, document: dict[str, object]) -> None:
        active_version = document.get("active_version")
        if not isinstance(active_version, int):
            return
        assets = self.repository.get_active_media_assets(str(document["id"]))
        needs_copy = [
            asset
            for asset in assets
            if not Path(str(asset["stored_path"]))
            .resolve()
            .is_relative_to(self.settings.published_media_dir.resolve())
        ]
        if not needs_copy:
            return
        paths = self._copy_media_snapshot(str(document["id"]), active_version, needs_copy)
        self.repository.update_active_media_paths(str(document["id"]), active_version, paths)

    def materialize_active_media_snapshots(self) -> None:
        for document in self.repository.list_documents():
            try:
                self._materialize_active_media_snapshot(document)
            except (OSError, ValueError):
                logger.exception(
                    "failed to materialize published media snapshot for %s",
                    document["id"],
                )

    async def ingest_path(
        self,
        source_path: Path,
        title: str | None = None,
        knowledge_domain: str = "company_public",
    ) -> dict[str, object]:
        extension = source_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {extension}")
        size_bytes = source_path.stat().st_size
        if size_bytes > self.settings.max_file_mb * 1024 * 1024:
            raise ValueError(f"文件超过 {self.settings.max_file_mb}MB 限制")

        sha256 = await asyncio.to_thread(file_sha256, source_path)
        existing = self.repository.find_document_by_sha256(sha256)
        if existing is not None:
            document = self.repository.get_document(str(existing["id"])) or existing
            if extension in {".pdf", ".xls", ".xlsx"} and not int(document.get("media_count", 0)):
                try:
                    await self.extract_media(str(existing["id"]))
                    document = self.repository.get_document(str(existing["id"])) or document
                except Exception:
                    logger.exception(
                        "media backfill failed for duplicate document %s", existing["id"]
                    )
            return {"document": document, "duplicate": True}

        document_id = str(uuid.uuid4())
        filename = safe_filename(source_path.name)
        destination_dir = self.settings.document_dir / document_id
        destination_dir.mkdir(parents=True, exist_ok=False)
        stored_path = destination_dir / filename
        await asyncio.to_thread(shutil.copy2, source_path, stored_path)

        now = utc_now()
        document = {
            "id": document_id,
            "title": title or source_path.stem,
            "filename": filename,
            "stored_path": str(stored_path),
            "file_type": extension.lstrip("."),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": "PROCESSING",
            "error_message": None,
            "knowledge_domain": knowledge_domain,
            "publication_status": "draft",
            "review_status": "pending",
            "visibility": "public",
            "created_at": now,
            "updated_at": now,
        }
        self.repository.create_document(document)
        try:
            parsed_chunks = await parse_document_isolated(
                stored_path,
                max_pdf_pages=self.settings.max_pdf_pages,
                ocr_timeout_seconds=self.settings.ocr_timeout_seconds,
                max_spreadsheet_rows=self.settings.max_spreadsheet_rows,
                max_archive_uncompressed_bytes=(
                    self.settings.max_archive_uncompressed_mb * 1024 * 1024
                ),
                max_archive_members=self.settings.max_archive_members,
            )
            if len(parsed_chunks) > self.settings.max_document_chunks:
                raise ValueError(f"文档分块超过 {self.settings.max_document_chunks} 个限制")
            await self._index_chunks(document_id, str(document["title"]), parsed_chunks)
            try:
                await self.extract_media(document_id)
            except Exception:
                logger.exception("media extraction failed for document %s", document_id)
            self.repository.update_document_status(document_id, "READY")
        except Exception as exc:
            self.repository.update_document_status(document_id, "FAILED", str(exc)[:1000])
            raise
        return {"document": self.repository.get_document(document_id), "duplicate": False}

    async def _index_chunks(
        self, document_id: str, title: str, parsed_chunks: list[ParsedChunk]
    ) -> None:
        embedding_inputs = [f"文档标题: {title}\n{chunk.content}" for chunk in parsed_chunks]
        embeddings = await self.ollama.embed(embedding_inputs)
        if len(embeddings) != len(parsed_chunks):
            raise RuntimeError("Embedding 返回数量与知识分块数量不一致")
        rows = []
        for position, (chunk, embedding) in enumerate(zip(parsed_chunks, embeddings, strict=True)):
            rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "position": position,
                    "content": chunk.content,
                    "page": chunk.page,
                    "sheet": chunk.sheet,
                    "row_start": chunk.row_start,
                    "row_end": chunk.row_end,
                    "embedding": pack_embedding(embedding),
                }
            )
        self.repository.replace_chunks(document_id, rows)

    async def extract_media_with_diff(
        self, document_id: str
    ) -> tuple[list[dict[str, object]], dict[str, int]]:
        document = self.repository.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        self._materialize_active_media_snapshot(document)
        source_path = Path(str(document["stored_path"]))
        if not source_path.exists():
            raise ValueError("原始文档文件不存在")
        output_dir = self.settings.media_dir / document_id
        extracted = await extract_document_media_isolated(
            source_path,
            output_dir,
            max_pdf_pages=self.settings.max_pdf_pages,
            max_archive_uncompressed_bytes=(
                self.settings.max_archive_uncompressed_mb * 1024 * 1024
            ),
            max_archive_members=self.settings.max_archive_members,
        )
        rows: list[dict[str, object]] = []
        for item in extracted:
            values = asdict(item)
            values["id"] = str(uuid.uuid4())
            values["product_ids"] = list(item.product_ids)
            values["enabled"] = True
            values["review_status"] = "pending"
            rows.append(values)
        mark_draft = document.get("active_version") is not None
        diff = self.repository.replace_media_assets(
            document_id,
            rows,
            mark_draft=mark_draft,
        )
        return self.repository.list_media_assets(document_id), diff

    async def extract_media(self, document_id: str) -> list[dict[str, object]]:
        assets, _diff = await self.extract_media_with_diff(document_id)
        return assets

    async def reindex_document(self, document_id: str) -> dict[str, object]:
        """Recompute vectors without reparsing or overwriting operator edits."""

        document = self.repository.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        chunks = self.repository.list_chunks(document_id)
        if not chunks:
            raise ValueError("文档没有可重算的知识片段")
        previous_status = str(document["status"])
        self.repository.update_document_status(document_id, "PROCESSING")
        try:
            inputs = [f"文档标题: {document['title']}\n{chunk['content']}" for chunk in chunks]
            embeddings = await self.ollama.embed(inputs)
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding 返回数量与知识片段数量不一致")
            self.repository.update_chunk_embeddings_draft(
                document_id,
                [
                    (str(chunk["id"]), embedding)
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
            self.repository.update_document_status(document_id, "READY")
        except Exception as exc:
            self.repository.update_document_status(document_id, previous_status, str(exc)[:1000])
            raise
        updated = self.repository.get_document(document_id)
        if updated is None:
            raise RuntimeError("重算向量后文档不存在")
        return updated

    def update_document(self, document_id: str, values: dict[str, object]) -> dict[str, object]:
        document = self.repository.get_document(document_id)
        if document is None:
            raise KeyError(document_id)
        if values.get("knowledge_domain") == "pet_care_general":
            raise ValueError("通用养宠知识使用本地模型自身知识，不接受文档上传")
        normalized = self._normalize_document_values(values)
        self._validate_effective_window(document, normalized)
        publication_status = normalized.pop("publication_status", None)
        normalized.pop("review_status", None)
        normalized.pop("reviewer_id", None)
        material_fields = {
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
        metadata = {key: value for key, value in normalized.items() if key in material_fields}

        if publication_status == "withdrawn":
            if metadata:
                raise ValueError("撤回与内容编辑请分开提交")
            self.repository.withdraw_document(document_id)
        elif publication_status == "published":
            prospective = {**document, **metadata}
            if prospective["status"] != "READY":
                raise ValueError("只有解析完成的文档才能发布")
            if prospective["visibility"] != "public":
                raise ValueError("公开问答只能发布 visibility=public 的文档")
            if prospective["owner_company_id"] != PUBLIC_OWNER_COMPANY_ID:
                raise ValueError("公开问答只能发布归属于本公司的资料")
            if not self.repository.list_chunks(document_id):
                raise ValueError("文档没有可发布的知识片段")
            changed_metadata = {
                key: value for key, value in metadata.items() if document.get(key) != value
            }
            target_version = int(document["source_version"]) + (1 if changed_metadata else 0)
            media_paths = self._copy_media_snapshot(
                document_id,
                target_version,
                self.repository.list_media_files(document_id),
            )
            self.repository.publish_document(
                document_id,
                reviewer_id="local-admin",
                immutable_media_paths=media_paths,
                metadata_updates=metadata,
            )
        else:
            changed_metadata = {
                key: value for key, value in metadata.items() if document.get(key) != value
            }
            if changed_metadata:
                self.repository.update_document_draft(document_id, changed_metadata)
        updated = self.repository.get_document(document_id)
        if updated is None:
            raise RuntimeError("更新后文档不存在")
        return updated

    async def update_chunk(
        self, chunk_id: str, content: str | None, enabled: bool | None
    ) -> dict[str, object]:
        chunk = self.repository.get_chunk(chunk_id)
        if chunk is None:
            raise KeyError(chunk_id)
        content_changed = content is not None and content != chunk["content"]
        enabled_changed = enabled is not None and bool(chunk["enabled"]) != enabled
        embedding: list[float] | None = None
        if content is not None and content_changed:
            embedding = (await self.ollama.embed([content]))[0]
        if content_changed or enabled_changed:
            self.repository.update_chunk_draft(
                chunk_id,
                content=content if content_changed else None,
                embedding=embedding,
                enabled=enabled if enabled_changed else None,
            )
        updated = self.repository.get_chunk(chunk_id)
        if updated is None:
            raise RuntimeError("知识片段更新后不存在")
        updated.pop("embedding", None)
        updated["enabled"] = bool(updated["enabled"])
        return updated

    def update_media_asset(self, asset_id: str, values: dict[str, object]) -> dict[str, object]:
        asset = self.repository.get_media_asset(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        editable = {"caption", "alt_text", "product_ids", "enabled"}
        changed = {
            key: value
            for key, value in values.items()
            if key in editable and asset.get(key) != value
        }
        if changed:
            self.repository.update_media_asset_draft(asset_id, changed)
        updated = self.repository.get_media_asset(asset_id)
        if updated is None:
            raise RuntimeError("图片资产更新后不存在")
        updated.pop("stored_path", None)
        updated.pop("search_text", None)
        return updated

    async def delete_document(self, document_id: str) -> bool:
        document = self.repository.get_document(document_id)
        if document is None:
            return False
        source_dir = Path(str(document["stored_path"])).parent
        media_dir = self.settings.media_dir / document_id
        published_media_dir = self.settings.published_media_dir / document_id
        destination: Path | None = None
        media_destination: Path | None = None
        published_media_destination: Path | None = None
        if source_dir.exists() or media_dir.exists() or published_media_dir.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            destination = self.settings.trash_dir / f"{stamp}-{document_id}"
            if source_dir.exists():
                await asyncio.to_thread(shutil.move, str(source_dir), str(destination))
            else:
                destination.mkdir(parents=True, exist_ok=False)
            if media_dir.exists():
                media_destination = destination / "media"
                await asyncio.to_thread(shutil.move, str(media_dir), str(media_destination))
            if published_media_dir.exists():
                published_media_destination = destination / "published_media"
                await asyncio.to_thread(
                    shutil.move,
                    str(published_media_dir),
                    str(published_media_destination),
                )
        try:
            deleted = self.repository.delete_document(document_id)
        except Exception:
            if published_media_destination is not None and published_media_destination.exists():
                await asyncio.to_thread(
                    shutil.move,
                    str(published_media_destination),
                    str(published_media_dir),
                )
            if media_destination is not None and media_destination.exists():
                await asyncio.to_thread(shutil.move, str(media_destination), str(media_dir))
            if destination is not None and destination.exists():
                await asyncio.to_thread(shutil.move, str(destination), str(source_dir))
            raise
        if deleted is None:
            if published_media_destination is not None and published_media_destination.exists():
                await asyncio.to_thread(
                    shutil.move,
                    str(published_media_destination),
                    str(published_media_dir),
                )
            if media_destination is not None and media_destination.exists():
                await asyncio.to_thread(shutil.move, str(media_destination), str(media_dir))
            if destination is not None and destination.exists():
                await asyncio.to_thread(shutil.move, str(destination), str(source_dir))
            return False
        return True


class AgentService:
    def __init__(
        self,
        settings: Settings,
        repository: KnowledgeRepository,
        ollama: OllamaClient,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.ollama = ollama
        self.boundary = AnswerBoundary(settings.boundary_policy_path)
        self.answer_cache = PublicAnswerCache(
            max_entries=settings.answer_cache_max_entries,
            ttl_seconds=settings.answer_cache_ttl_seconds,
        )

    def _public_cache_key(
        self,
        request: ChatRequest,
        decision: BoundaryDecision,
        safe_question: str,
        content_mode: str,
    ) -> str | None:
        if (
            not self.settings.answer_cache_enabled
            or request.history
            or decision.mode != "retrieve"
            or decision.partial_notice
            or content_mode != "text_only"
            or _PERSONAL_OR_VOLATILE_QUESTION_PATTERN.search(request.question)
        ):
            return None
        normalized = " ".join(safe_question.strip().split()).casefold()
        revision = self.repository.public_knowledge_revision(decision.domains)
        value = json.dumps(
            {
                "question": normalized,
                "domains": decision.domains,
                "reason": decision.reason_code,
                "policy": decision.policy_version,
                "model": self.settings.chat_model,
                "prompt": PROMPT_VERSION,
                "revision": revision,
                "scope": "public",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _cached_sources_active(self, response: ChatResponse) -> bool:
        if not response.sources or response.images:
            return False
        snapshots: list[tuple[str, int, str]] = []
        for source in response.sources:
            if source.source_kind != "text" or source.source_version is None:
                return False
            snapshots.append((source.document_id, source.source_version, source.chunk_id))
        return self.repository.are_source_snapshots_active(snapshots)

    @asynccontextmanager
    async def _model_stage_budget(
        self,
        deadline: float | None,
        stage: str,
        *,
        cap_seconds: float | None = None,
    ) -> AsyncIterator[None]:
        if deadline is None:
            yield
            return
        loop = asyncio.get_running_loop()
        stage_deadline = deadline - self.settings.chat_finalize_reserve_seconds
        if cap_seconds is not None:
            stage_deadline = min(stage_deadline, loop.time() + cap_seconds)
        if stage_deadline <= loop.time():
            raise ModelStageBudgetExceeded(stage)
        try:
            async with asyncio.timeout_at(stage_deadline):
                yield
        except TimeoutError as exc:
            raise ModelStageBudgetExceeded(stage) from exc

    @staticmethod
    def _is_image_request(question: str) -> bool:
        return _IMAGE_REQUEST_PATTERN.search(question) is not None

    @classmethod
    def _response_content_mode(cls, question: str) -> str:
        """Distinguish image-only, text-only, and combined requests."""

        wants_images = cls._is_image_request(question)
        wants_text = _TEXT_DETAIL_REQUEST_PATTERN.search(question) is not None
        if wants_images and wants_text:
            return "text_and_image"
        if wants_images:
            return "image_only"
        return "text_only"

    @staticmethod
    def _generation_question(question: str) -> str:
        if not _BROAD_BUSINESS_PATTERN.search(question):
            return question
        if _EXPLICIT_BUSINESS_DETAIL_PATTERN.search(question):
            return question
        return (
            f"{question}\n\n"
            "回答要求：只概括主营业务、产品或服务范围、制造工艺和销售范围。"
            "用户没有询问经营规模，不主动展开面积、数量、比例、金额、专利数量、"
            "中心或分公司数量等数字。"
        )

    def _self_only_question(self, question: str, decision: BoundaryDecision) -> str:
        """Rewrite a mixed-brand question so the model sees only the allowed subject."""
        if not decision.partial_notice:
            return question

        owned_name = ""
        for alias in sorted(self.boundary.owned_aliases, key=len, reverse=True):
            match = re.search(re.escape(alias), question, flags=re.IGNORECASE)
            if match:
                owned_name = match.group(0)
                break

        if "product_public" in decision.domains:
            product_types = list(dict.fromkeys(_PRODUCT_TYPE_PATTERN.findall(question)))
            product_type = product_types[0] if len(product_types) == 1 else ""
            models = list(
                dict.fromkeys(
                    value.replace(" ", "").upper()
                    for value in _MEDIA_CODE_PATTERN.findall(question.upper())
                )
            )
            variants = list(
                dict.fromkeys(
                    match.group(0) for match in _PRODUCT_VARIANT_PATTERN.finditer(question)
                )
            )
            aspects = list(dict.fromkeys(_REQUEST_ASPECT_PATTERN.findall(question)))
            subject = owned_name or "本公司自有品牌"
            target = " ".join(
                value
                for value in (
                    subject,
                    "、".join(models),
                    "、".join(variants),
                    product_type or "自有产品",
                )
                if value
            )
            requested_content = "、".join(aspects) if aspects else "型号、功能和适用场景"
            excluded_scope = (
                "不要回答疾病诊断、治疗或用药部分。"
                if decision.partial_kind == "medical"
                else "不要提及、评价或比较任何其他品牌。"
            )
            return f"请仅依据已审核资料回答{target}的{requested_content}。{excluded_scope}"

        subject = owned_name or "本公司"
        excluded_scope = (
            "不要回答疾病诊断、治疗或用药部分。"
            if decision.partial_kind == "medical"
            else "不要提及、评价或比较任何其他公司或品牌。"
        )
        return f"请仅依据已审核资料介绍{subject}的公开业务、品牌和优势。{excluded_scope}"

    @staticmethod
    def _self_only_follow_ups(decision: BoundaryDecision) -> list[str]:
        if decision.follow_up_questions:
            return list(decision.follow_up_questions)
        if "product_public" in decision.domains:
            return ["请介绍自有产品的具体型号", "请展示自有产品图片"]
        return ["请介绍公司的核心业务", "请介绍自有品牌与产品"]

    def _sanitized_history(self, history: list[ChatMessage]) -> list[ChatMessage]:
        """Keep context while preventing third-party brand names from reaching the model."""

        sanitized: list[ChatMessage] = []
        for message in history:
            content = message.content
            for alias in self.boundary.blocked_third_party_aliases:
                content = re.sub(
                    re.escape(alias),
                    "[第三方品牌]",
                    content,
                    flags=re.IGNORECASE,
                )
            sanitized.append(message.model_copy(update={"content": content}))
        return sanitized

    @staticmethod
    def _question_task(text: str) -> str:
        return next((name for name, pattern in _TASK_PATTERNS if pattern.search(text)), "")

    def _history_state(self, history: list[ChatMessage]) -> ConversationState:
        """Extract minimal recent state; history is never treated as factual evidence."""

        recent = history[-6:]
        for index in range(len(recent) - 1, -1, -1):
            message = recent[index]
            if message.role != "user":
                continue
            user_text = message.content
            companion = ""
            if index + 1 < len(recent) and recent[index + 1].role == "assistant":
                companion = recent[index + 1].content
            context = f"{user_text}\n{companion}"

            third_parties = self.boundary._third_party_mentions(  # noqa: SLF001
                user_text,
                self.boundary.blocked_third_party_aliases,
            )
            categories = list(dict.fromkeys(_PRODUCT_TYPE_PATTERN.findall(context)))
            category = categories[0] if len(categories) == 1 else ""
            if third_parties:
                subject = " ".join((third_parties[0], category)).strip()
                return ConversationState(
                    subject=subject,
                    topic="product",
                    category=category,
                    task=self._question_task(user_text),
                    authorized=False,
                )

            codes = list(
                dict.fromkeys(
                    value.replace(" ", "").upper()
                    for value in _MEDIA_CODE_PATTERN.findall(context.upper())
                )
            )
            owned_alias = next(
                (
                    alias
                    for alias in sorted(self.boundary.owned_aliases, key=len, reverse=True)
                    if alias in context.lower()
                ),
                "",
            )
            if len(codes) > 1 or len(categories) > 1:
                return ConversationState(ambiguous=True)
            if codes:
                variant_match = _PRODUCT_VARIANT_PATTERN.search(context)
                variant = variant_match.group(0) if variant_match else ""
                brand = owned_alias or "CKIN PET"
                subject = " ".join((brand, codes[0], variant, category)).strip()
                return ConversationState(
                    subject=subject,
                    topic="product",
                    brand=brand,
                    model=codes[0],
                    category=category,
                    variant=variant,
                    task=self._question_task(user_text),
                )
            if owned_alias and category:
                variant_match = _PRODUCT_VARIANT_PATTERN.search(context)
                variant = variant_match.group(0) if variant_match else ""
                subject = " ".join((owned_alias, variant, category)).strip()
                return ConversationState(
                    subject=subject,
                    topic="product",
                    brand=owned_alias,
                    category=category,
                    variant=variant,
                    task=self._question_task(user_text),
                )

            # A pet subject may safely resolve a later pronoun, but it must come
            # from the user's own message. Assistant text is context, not fact.
            pet_mentions: list[str] = []
            lowered_user_text = user_text.lower()
            for term in sorted(self.boundary.pet_subjects, key=len, reverse=True):
                if term not in lowered_user_text:
                    continue
                if any(term in existing for existing in pet_mentions):
                    continue
                pet_mentions.append(term)

            specific_pet_mentions = [
                term for term in pet_mentions if term not in _GENERIC_PET_SUBJECTS
            ]
            if len(specific_pet_mentions) > 1:
                return ConversationState(topic="pet_care", ambiguous=True)
            if specific_pet_mentions:
                return ConversationState(
                    subject=specific_pet_mentions[0],
                    topic="pet_care",
                    task=self._question_task(user_text),
                )
            if len(pet_mentions) > 1:
                return ConversationState(topic="pet_care", ambiguous=True)
            if pet_mentions:
                return ConversationState(
                    subject=pet_mentions[0],
                    topic="pet_care",
                    task=self._question_task(user_text),
                )
        return ConversationState()

    def _recent_pet_context(self, history: list[ChatMessage]) -> list[ChatMessage]:
        """Keep only turns after the latest explicit change away from pet care."""

        recent = history[-6:]
        cutoff = 0
        for index, message in enumerate(recent):
            if message.role != "user":
                continue
            decision = self.boundary.classify(message.content)
            if (
                decision.mode == "retrieve"
                or decision.reason_code
                in {
                    "THIRD_PARTY_BRAND",
                    "UNRELATED_TOPIC",
                    "PET_MEDICAL_ADVICE",
                    "MEDICAL_EMERGENCY",
                    "PRIVATE_OR_NONPUBLIC_INFORMATION",
                    "PRODUCT_IDENTITY_AMBIGUOUS",
                }
                or _NON_PET_CONTEXT_BREAK_PATTERN.search(message.content)
            ):
                cutoff = index + 1
        return recent[cutoff:]

    def resolve_question(self, request: ChatRequest) -> str:
        """Resolve safe references and elliptical follow-ups from minimal recent state."""

        question = request.question
        has_reference = _DEMONSTRATIVE_PATTERN.search(question) is not None
        is_follow_up = _FOLLOWUP_PATTERN.search(question.strip()) is not None
        needs_pet_subject = self.boundary.needs_pet_subject_context(question)
        current_facets = extract_product_facets(question)
        only_modifier = bool(current_facets.variants) and not (
            current_facets.models or current_facets.categories
        )
        if not has_reference and not is_follow_up and not only_modifier and not needs_pet_subject:
            return question

        raw_decision = self.boundary.classify(question)
        if raw_decision.reason_code in {
            "PET_CARE_GENERAL_MODEL_KNOWLEDGE",
            "PET_MEDICAL_ADVICE",
            "MEDICAL_EMERGENCY",
            "PRIVATE_OR_NONPUBLIC_INFORMATION",
            "UNRELATED_TOPIC",
            "PERSONALIZED_FEEDING_AMOUNT",
            "CONVERSATIONAL",
        } and not (raw_decision.verify_topic and (has_reference or is_follow_up)):
            return question
        if raw_decision.reason_code == "THIRD_PARTY_BRAND" and not has_reference:
            return question

        state = self._history_state(request.history)
        if state.topic == "pet_care":
            state = self._history_state(self._recent_pet_context(request.history))
        if state.ambiguous or not state.subject:
            return question

        if needs_pet_subject:
            if state.topic != "pet_care":
                return question
            if has_reference:
                return _CONTEXT_REFERENCE_PATTERN.sub(state.subject, question)
            return f"{state.subject}{question}"

        if has_reference:
            return _CONTEXT_REFERENCE_PATTERN.sub(state.subject, question)

        if state.topic == "pet_care":
            return question

        if current_facets.categories and not current_facets.models:
            return question

        subject = state.subject
        if current_facets.models:
            requested_model = sorted(current_facets.models)[0]
            subject = " ".join(
                value
                for value in (state.brand or "CKIN PET", requested_model, state.category)
                if value
            )
        suffix = question
        current_task = self._question_task(question)
        if not current_task and state.task:
            suffix = f"{question}，继续询问{state.task}"
        return f"关于{subject}，{suffix}"

    @staticmethod
    def _ensure_requested_model_coverage(
        question: str,
        query_embedding: list[float],
        chunks: list[dict[str, object]],
        results: list[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Prioritize one directly matching chunk per model in multi-model questions."""

        query_codes = list(
            dict.fromkeys(
                value.replace(" ", "").upper()
                for value in _MEDIA_CODE_PATTERN.findall(question.upper())
            )
        )
        if len(query_codes) < 2:
            return results

        coverage: list[SearchResult] = []
        for code in query_codes:
            existing = next(
                (result for result in results if code in result.content.upper().replace(" ", "")),
                None,
            )
            if existing is None:
                targeted = retrieve(code, query_embedding, chunks, top_k=top_k)
                existing = next(
                    (
                        result
                        for result in targeted
                        if code in result.content.upper().replace(" ", "")
                    ),
                    None,
                )
            if existing is not None and all(
                existing.source.chunk_id != item.source.chunk_id for item in coverage
            ):
                coverage.append(existing)

        merged = coverage + [
            result
            for result in results
            if all(result.source.chunk_id != item.source.chunk_id for item in coverage)
        ]
        return [
            SearchResult(
                source=result.source.model_copy(update={"source_id": f"S{index}"}),
                content=result.content,
            )
            for index, result in enumerate(merged[:top_k], start=1)
        ]

    @staticmethod
    def _has_capacity_conflict(question: str, results: list[SearchResult]) -> bool:
        """Detect incompatible capacity values bound to the same requested product."""

        if "容量" not in question and "容积" not in question:
            return False
        query_codes = {
            value.replace(" ", "").upper()
            for value in _MEDIA_CODE_PATTERN.findall(question.upper())
        }
        query_categories = set(_PRODUCT_TYPE_PATTERN.findall(question))
        values_by_subject: dict[tuple[str, str], set[Decimal]] = {}

        for result in results:
            content = result.content
            content_upper = content.upper().replace(" ", "")
            matching_codes = {
                code for code in query_codes if code.replace(" ", "") in content_upper
            }
            if query_codes and not matching_codes:
                continue
            content_categories = set(_PRODUCT_TYPE_PATTERN.findall(content))
            if query_categories and not (query_categories & content_categories):
                continue

            for match in _CAPACITY_VALUE_PATTERN.finditer(content):
                window = content[max(0, match.start() - 100) : match.end() + 80]
                if not re.search(r"容量|容积|水箱|粮仓|储粮", window):
                    continue
                local_categories = set(_PRODUCT_TYPE_PATTERN.findall(window))
                categories = query_categories & local_categories
                if query_categories and not categories:
                    categories = query_categories & content_categories
                if not categories:
                    categories = local_categories or content_categories or {""}
                codes = matching_codes or {""}
                multiplier = (
                    Decimal("1000") if match.group("unit").lower() in {"l", "升"} else Decimal("1")
                )
                normalized = Decimal(match.group("value")) * multiplier
                for code in codes:
                    for category in categories:
                        values_by_subject.setdefault((code, category), set()).add(normalized)

        return any(len(values) > 1 for values in values_by_subject.values())

    @staticmethod
    def _missing_strict_evidence(question: str, results: list[SearchResult]) -> bool:
        """Require direct facts for volatile, exhaustive, or precision-sensitive requests."""

        return any(
            question_pattern.search(question)
            and not any(evidence_pattern.search(result.content) for result in results)
            for question_pattern, evidence_pattern in _STRICT_EVIDENCE_REQUIREMENTS
        )

    @staticmethod
    def _salvage_covers_requested_facts(question: str, answer: GeneratedAnswer) -> bool:
        """Do not use a shortened draft if it lost the requested product or attribute."""

        answer_text = "\n".join(
            [answer.title, answer.summary, *(section.content for section in answer.sections)]
        )
        requested = extract_product_facets(question)
        remaining = extract_product_facets(answer_text)
        if any(
            not expected.issubset(actual)
            for expected, actual in (
                (requested.brands, remaining.brands),
                (requested.categories, remaining.categories),
                (requested.models, remaining.models),
                (requested.variants, remaining.variants),
            )
        ):
            return False
        return all(
            not question_pattern.search(question) or evidence_pattern.search(answer_text)
            for question_pattern, evidence_pattern in _DIRECT_EVIDENCE_REQUIREMENTS
        )

    def _extractive_fallback(
        self,
        question: str,
        decision: BoundaryDecision,
        results: list[SearchResult],
        guard_reason: str,
    ) -> tuple[GeneratedAnswer, GuardResult] | None:
        """Build a cited answer only from verbatim published evidence after generation fails."""

        if guard_reason not in _EXTRACTIVE_FALLBACK_REASONS:
            return None
        if _EXTRACTIVE_SKIP_PATTERN.search(question):
            return None

        query_codes = tuple(
            dict.fromkeys(
                value.replace(" ", "").upper()
                for value in _MEDIA_CODE_PATTERN.findall(question.upper())
            )
        )
        query_categories = tuple(dict.fromkeys(_PRODUCT_TYPE_PATTERN.findall(question)))
        query_variants = tuple(
            dict.fromkeys(match.group(0) for match in _PRODUCT_VARIANT_PATTERN.finditer(question))
        )
        query_aliases = tuple(
            dict.fromkeys(
                alias
                for alias in sorted(self.boundary.owned_aliases, key=len, reverse=True)
                if alias in question.lower()
            )
        )

        eligible_results = list(results)
        required_evidence_patterns: list[re.Pattern[str]] = []
        for question_pattern, evidence_pattern in _DIRECT_EVIDENCE_REQUIREMENTS:
            if not question_pattern.search(question):
                continue
            required_evidence_patterns.append(evidence_pattern)
            eligible_results = [
                result for result in eligible_results if evidence_pattern.search(result.content)
            ]
            if not eligible_results:
                return None

        combined = "\n".join(result.content for result in eligible_results)
        combined_folded = combined.lower().replace(" ", "")
        product_anchors = (*query_codes, *query_variants, *query_categories)
        required_anchors = product_anchors or query_aliases
        for anchor in required_anchors:
            if anchor.lower().replace(" ", "") not in combined_folded:
                return None

        candidates: list[tuple[float, SearchResult, str]] = []
        for result in eligible_results:
            for anchor in (*query_codes, *query_variants):
                anchor_match = re.search(
                    re.escape(anchor),
                    result.content,
                    flags=re.IGNORECASE,
                )
                if anchor_match is None:
                    continue
                position = anchor_match.start()
                raw_window = result.content[position : position + 900]
                if len(raw_window) == 900:
                    last_boundary = max(
                        (raw_window.rfind(mark) for mark in "。！？!?;；\n"),
                        default=-1,
                    )
                    if last_boundary >= 300:
                        raw_window = raw_window[: last_boundary + 1]
                segment = " ".join(raw_window.split()).strip()
                output_check = validate_public_output_text(
                    segment,
                    self.boundary.blocked_third_party_aliases,
                )
                if output_check.valid:
                    candidates.append((lexical_score(question, segment) + 0.2, result, segment))
            raw_segments = _EXTRACTIVE_SEGMENT_SPLIT_PATTERN.split(result.content)
            for raw_segment in raw_segments:
                segment = " ".join(raw_segment.split()).strip()
                substantive = re.sub(r"^[#*\-\d.、：:\s]+", "", segment)
                if len(substantive) < 8 or len(segment) > 1100:
                    continue
                if re.fullmatch(r"第\d+列\s*[:：]\s*[^。！？!?；;]{1,100}", segment):
                    continue
                if segment.startswith("#") or segment in {"产品特点", "产品信息", "功能"}:
                    continue
                output_check = validate_public_output_text(
                    segment,
                    self.boundary.blocked_third_party_aliases,
                )
                if not output_check.valid:
                    continue
                score = lexical_score(question, segment)
                if score >= 0.08 or any(
                    anchor.lower().replace(" ", "") in segment.lower().replace(" ", "")
                    for anchor in required_anchors
                ):
                    candidates.append((score, result, segment))
        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
        if any(
            not any(
                anchor.lower().replace(" ", "") in candidate[2].lower().replace(" ", "")
                for candidate in candidates
            )
            for anchor in required_anchors
        ):
            return None
        selected: list[tuple[float, SearchResult, str]] = []
        for evidence_pattern in required_evidence_patterns:
            match = next(
                (candidate for candidate in candidates if evidence_pattern.search(candidate[2])),
                None,
            )
            if match is not None and all(match[2] != item[2] for item in selected):
                selected.append(match)
        for anchor in required_anchors:
            normalized_anchor = anchor.lower().replace(" ", "")
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if normalized_anchor in candidate[2].lower().replace(" ", "")
                ),
                None,
            )
            if match is not None and all(match[2] != item[2] for item in selected):
                selected.append(match)
        if not selected:
            selected.extend(candidates[:2])
        if not selected:
            return None

        selected = selected[:5]
        first = selected[0]
        generated_intent = (
            decision.intent
            if decision.intent
            in {"company_intro", "product_intro", "product_comparison", "faq", "other"}
            else "other"
        )
        generated = GeneratedAnswer(
            status="grounded",
            intent=generated_intent,  # type: ignore[arg-type]
            title="已审核资料中的相关信息",
            summary=first[2],
            summary_source_ids=[first[1].source.source_id],
            sections=[
                GeneratedSection(
                    heading="资料原文",
                    content=segment,
                    source_ids=[result.source.source_id],
                )
                for _score, result, segment in selected
            ],
            follow_up_questions=[],
        )
        fallback_guard = validate_generated_answer(
            generated,
            results,
            self.boundary.blocked_third_party_aliases,
        )
        if not fallback_guard.valid:
            return None
        return generated, fallback_guard

    @staticmethod
    def _media_query_text(request: ChatRequest) -> str:
        if not _DEMONSTRATIVE_PATTERN.search(request.question):
            return request.question
        recent = "\n".join(message.content for message in request.history[-4:])
        return f"{recent}\n{request.question}"[-16_000:]

    def _source_snapshots_are_active(
        self,
        results: list[SearchResult],
        source_ids: set[str],
    ) -> bool:
        selected = [result for result in results if result.source.source_id in source_ids]
        if not selected:
            return False
        snapshots: list[tuple[str, int, str]] = []
        for result in selected:
            version = result.source.source_version
            if version is None:
                return False
            snapshots.append((result.source.document_id, version, result.source.chunk_id))
        return self.repository.are_source_snapshots_active(snapshots)

    @staticmethod
    def _location_score(asset: dict[str, object], result: SearchResult) -> float:
        source = result.source
        if str(asset["document_id"]) != source.document_id:
            return -100.0
        score = 2.0
        asset_page = asset.get("page")
        if isinstance(asset_page, int) and source.page == asset_page:
            score += 20.0
        asset_sheet = asset.get("sheet")
        if isinstance(asset_sheet, str) and source.sheet == asset_sheet:
            score += 3.0
            asset_start = asset.get("row_start")
            asset_end = asset.get("row_end")
            source_start = source.row_start
            source_end = source.row_end or source_start
            if (
                isinstance(asset_start, int)
                and isinstance(asset_end, int)
                and source_start is not None
                and source_end is not None
                and asset_start <= source_end
                and asset_end >= source_start
            ):
                score += 20.0
        return score

    def _select_response_images(
        self,
        request: ChatRequest,
        results: list[SearchResult],
        *,
        limit: int = 6,
    ) -> list[ResponseImage]:
        document_ids = list(dict.fromkeys(result.source.document_id for result in results))
        assets = self.repository.fetch_public_media(document_ids or None)
        if not assets:
            return []

        query_text = self._media_query_text(request)
        query_lower = query_text.lower()
        query_facets = extract_product_facets(query_text)
        if results and not query_facets.categories:
            evidence_facets = extract_product_facets(
                "\n".join(result.content for result in results)
            )
            if len(evidence_facets.categories) == 1:
                # A model code such as F02 can exist in several categories. For a
                # combined text-and-image response, follow the category actually
                # supported by the final cited text rather than mixing same-code
                # images from another product family.
                query_facets = ProductFacets(
                    brands=query_facets.brands,
                    categories=evidence_facets.categories,
                    models=query_facets.models,
                    variants=query_facets.variants,
                )
        query_codes = {
            value.replace(" ", "").upper()
            for value in _MEDIA_CODE_PATTERN.findall(query_text.upper())
        }
        query_keywords = {keyword for keyword in _MEDIA_KEYWORDS if keyword in query_lower}
        scored: list[tuple[float, dict[str, object]]] = []
        for asset in assets:
            caption = str(asset.get("caption") or "产品资料图")
            alt_text = str(asset.get("alt_text") or caption)
            output_check = validate_public_output_text(
                f"{caption}\n{alt_text}",
                self.boundary.blocked_third_party_aliases,
            )
            if not output_check.valid:
                continue
            searchable = "\n".join(
                (
                    caption,
                    alt_text,
                    str(asset.get("search_text", "")),
                    str(asset.get("document_title", "")),
                    " ".join(str(value) for value in asset.get("product_ids", [])),
                )
            )
            searchable_lower = searchable.lower()
            candidate_facets = extract_product_facets(searchable)
            if not product_facets_match(query_facets, candidate_facets):
                continue
            lexical = lexical_score(query_text, searchable) * 10.0
            codes = {
                value.replace(" ", "").upper()
                for value in _MEDIA_CODE_PATTERN.findall(searchable.upper())
            }
            if query_codes and not (query_codes & codes):
                continue
            code_score = 0.0
            if query_codes:
                code_score = 14.0 * len(query_codes & codes)
            keyword_score = 4.0 * sum(
                1 for keyword in query_keywords if keyword in searchable_lower
            )
            best_location = -100.0
            best_source_id: str | None = None
            for result in results:
                location = self._location_score(asset, result)
                if location > best_location:
                    best_location = location
                    best_source_id = result.source.source_id
            kind_bonus = 2.0 if asset.get("source_kind") == "product_image" else 0.0
            identity_score = 8.0 * sum(
                bool(expected and expected & actual)
                for expected, actual in (
                    (query_facets.models, candidate_facets.models),
                    (query_facets.categories, candidate_facets.categories),
                    (query_facets.variants, candidate_facets.variants),
                )
            )
            location_score = max(best_location, 0.0) if best_source_id else 0.0
            score = lexical + code_score + keyword_score + identity_score + location_score
            score += kind_bonus
            if score > 1.0:
                scored.append((score, asset))

        scored.sort(
            key=lambda item: (
                item[0],
                int(str(item[1].get("width", 0))) * int(str(item[1].get("height", 0))),
            ),
            reverse=True,
        )
        selected: list[ResponseImage] = []
        seen_hashes: set[str] = set()
        for _score, asset in scored:
            digest = str(asset["sha256"])
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            caption = str(asset.get("caption") or "产品资料图")
            alt_text = str(asset.get("alt_text") or caption)
            raw_product_ids = asset.get("product_ids")
            product_ids = (
                [str(value) for value in raw_product_ids]
                if isinstance(raw_product_ids, list)
                else []
            )
            raw_page = asset.get("page")
            raw_row_start = asset.get("row_start")
            raw_row_end = asset.get("row_end")
            search_text = str(asset.get("search_text", ""))
            product_keys = [
                identity.stable_id
                for identity in extract_product_identities(
                    "\n".join((caption, alt_text, search_text))
                )
            ]
            source_version = asset.get("source_version")
            selected.append(
                ResponseImage(
                    id=str(asset["id"]),
                    url=f"/api/media/{asset['id']}",
                    alt_text=alt_text,
                    caption=caption,
                    source_id=f"M{len(selected) + 1}",
                    document_id=str(asset["document_id"]),
                    document_title=str(asset["document_title"]),
                    document_filename=str(asset.get("document_filename") or ""),
                    product_ids=product_ids,
                    product_keys=product_keys,
                    source_version=(source_version if isinstance(source_version, int) else None),
                    page=raw_page if isinstance(raw_page, int) else None,
                    sheet=(str(asset["sheet"]) if isinstance(asset.get("sheet"), str) else None),
                    row_start=raw_row_start if isinstance(raw_row_start, int) else None,
                    row_end=raw_row_end if isinstance(raw_row_end, int) else None,
                    width=int(str(asset["width"])),
                    height=int(str(asset["height"])),
                )
            )
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _media_source_refs(images: list[ResponseImage]) -> list[SourceRef]:
        sources: list[SourceRef] = []
        for image in images:
            if image.source_id is None:
                continue
            excerpt = image.caption or image.alt_text
            sources.append(
                SourceRef(
                    source_id=image.source_id,
                    chunk_id=f"media:{image.id}",
                    document_id=image.document_id,
                    document_title=image.document_title,
                    filename=image.document_filename,
                    page=image.page,
                    sheet=image.sheet,
                    row_start=image.row_start,
                    row_end=image.row_end,
                    score=1.0,
                    excerpt=excerpt,
                    source_version=image.source_version,
                    source_kind="media",
                )
            )
        return sources

    def _media_snapshots_are_active(self, images: list[ResponseImage]) -> bool:
        snapshots = [
            (image.document_id, image.source_version, image.id)
            for image in images
            if image.source_version is not None
        ]
        return len(snapshots) == len(images) and self.repository.are_media_snapshots_active(
            [(document_id, version, asset_id) for document_id, version, asset_id in snapshots]
        )

    def _image_response(
        self,
        request_id: str,
        request: ChatRequest,
        decision: BoundaryDecision,
        results: list[SearchResult],
        images: list[ResponseImage],
        retrieval_ms: int,
    ) -> ChatResponse:
        sources = self._media_source_refs(images)
        cited_ids = [source.source_id for source in sources]
        markers = "".join(f"[{source_id}]" for source_id in cited_ids)
        image_summary = f"知识库中找到 {len(images)} 张与问题匹配的已审核产品图片。{markers}"
        sections = [
            AnswerSection(
                heading="图片说明",
                content="图片直接提取自已审核发布的原始资料；标题下方标注了文档位置。",
            )
        ]
        summary = image_summary
        follow_up_questions = ["请介绍图片中的产品功能", "查看其他产品图片"]
        if decision.partial_notice:
            summary = decision.partial_notice
            sections.insert(0, AnswerSection(heading="自有产品图片", content=image_summary))
            follow_up_questions = self._self_only_follow_ups(decision)
        evidence_ids = [source.source_id for source in sources]
        response = ChatResponse(
            request_id=request_id,
            status=decision.status,
            intent=decision.intent,
            evidence_status=decision.evidence_status,
            decision=DecisionView(
                reason_code="PRODUCT_MEDIA_FOUND",
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title="产品图片",
                summary=summary,
                refusal_reason=decision.refusal_reason,
                sections=sections,
            ),
            images=images,
            thinking_summary=[
                ThinkingStep(
                    stage="intent",
                    summary="识别到用户希望查看产品图片。",
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="retrieval",
                    summary=(
                        f"优先查询已审核媒体目录，并按品牌、品类、型号、款式匹配到 "
                        f"{len(images)} 张图片。"
                    ),
                    evidence_ids=evidence_ids,
                ),
                ThinkingStep(
                    stage="evidence",
                    summary="仅展示已审核发布且实际存在于原始文档中的图片。",
                    evidence_ids=evidence_ids,
                ),
                ThinkingStep(
                    stage="conclusion",
                    summary="图片资产与来源位置校验通过，无需调用生成模型猜测图片内容。",
                    evidence_ids=evidence_ids,
                ),
            ],
            sources=sources,
            confidence=None,
            follow_up_questions=follow_up_questions,
            metrics={
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0,
                "retrieved_chunks": len(results),
            },
        )
        return response

    def _record(self, question: str, response: ChatResponse) -> None:
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        self.repository.log_query(
            {
                "id": response.request_id,
                "question": question_hash,
                "status": response.status,
                "source_ids": json.dumps(
                    [source.source_id for source in response.sources], ensure_ascii=False
                ),
                "source_snapshots": json.dumps(
                    [
                        {
                            "document_id": source.document_id,
                            "chunk_id": source.chunk_id,
                            "version": source.source_version,
                            "kind": source.source_kind,
                        }
                        for source in response.sources
                    ],
                    ensure_ascii=False,
                ),
                "retrieval_ms": int(response.metrics.get("retrieval_ms", 0)),
                "generation_ms": int(response.metrics.get("generation_ms", 0)),
                "topic_check_ms": int(response.metrics.get("topic_check_ms", 0)),
                "initial_generation_ms": int(response.metrics.get("initial_generation_ms", 0)),
                "repair_ms": int(response.metrics.get("repair_ms", 0)),
                "repair_attempts": int(response.metrics.get("repair_attempts", 0)),
                "repair_reason": str(response.metrics.get("repair_reason", "")) or None,
                "queue_ms": int(response.metrics.get("queue_ms", 0)),
                "total_ms": int(response.metrics.get("total_ms", 0)),
                "cache_hit": int(response.metrics.get("cache_hit", 0)),
                "model_name": self.settings.chat_model,
                "prompt_version": PROMPT_VERSION,
                "reason_code": response.decision.reason_code,
                "policy_version": response.decision.policy_version,
                "evidence_status": response.evidence_status,
                "fallback_kind": response.fallback_kind,
                "created_at": utc_now(),
            }
        )

    def _fixed_response(self, request_id: str, decision: BoundaryDecision) -> ChatResponse:
        fallback_kind: FallbackKind | None = None
        if decision.status == "out_of_scope":
            fallback_kind = "policy_boundary"
        elif decision.status == "clarification_needed":
            fallback_kind = "clarification"
        elif decision.status == "service_unavailable":
            fallback_kind = "service_error"
        response = ChatResponse(
            request_id=request_id,
            status=decision.status,
            intent=decision.intent,
            evidence_status=decision.evidence_status,
            fallback_kind=fallback_kind,
            decision=DecisionView(
                reason_code=decision.reason_code,
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title=decision.title,
                summary=decision.summary,
                refusal_reason=decision.refusal_reason,
                sections=[],
            ),
            thinking_summary=[
                ThinkingStep(
                    stage="intent",
                    summary="先按服务范围与安全边界判断当前问题。",
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="conclusion",
                    summary="该场景使用服务端固定策略答复，未调用知识检索或生成模型。",
                    evidence_ids=[],
                ),
            ],
            sources=[],
            confidence=None,
            follow_up_questions=list(decision.follow_up_questions),
            metrics={"retrieval_ms": 0, "generation_ms": 0, "retrieved_chunks": 0},
        )
        return response

    def _insufficient_response(
        self,
        request_id: str,
        decision: BoundaryDecision,
        message: str,
        *,
        reason_code: str,
        retrieval_ms: int = 0,
        generation_ms: int = 0,
        retrieved_chunks: int = 0,
    ) -> ChatResponse:
        steps = [
            ThinkingStep(
                stage="intent",
                summary="问题属于允许回答的范围。",
                evidence_ids=[],
            )
        ]
        if retrieved_chunks:
            steps.append(
                ThinkingStep(
                    stage="retrieval",
                    summary=f"检索到 {retrieved_chunks} 个候选片段，但未形成可验证答案。",
                    evidence_ids=[],
                )
            )
        steps.append(ThinkingStep(stage="evidence", summary=message, evidence_ids=[]))
        title = "暂时无法确认"
        refusal_reason = message
        if decision.partial_notice:
            template_name = (
                "mixed_medical" if decision.partial_kind == "medical" else "mixed_third_party"
            )
            title = self.boundary.templates[template_name]["title"]
            message = f"{decision.partial_notice} {message}"
            if decision.refusal_reason:
                refusal_reason = f"{decision.refusal_reason}另外，{refusal_reason}"
        return ChatResponse(
            request_id=request_id,
            status="insufficient_evidence",
            intent=decision.intent,
            evidence_status="missing",
            fallback_kind="missing_evidence",
            decision=DecisionView(
                reason_code=reason_code,
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title=title,
                summary=message,
                refusal_reason=refusal_reason,
                sections=[],
            ),
            thinking_summary=steps,
            sources=[],
            confidence=None,
            follow_up_questions=["请补充经审核的相关资料", "换一种更具体的问法"],
            metrics={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "retrieved_chunks": retrieved_chunks,
            },
        )

    def _self_only_safe_fallback(
        self,
        request_id: str,
        decision: BoundaryDecision,
        *,
        reason_code: str,
        retrieval_ms: int,
        generation_ms: int,
        retrieved_chunks: int,
    ) -> ChatResponse:
        template_name = (
            "mixed_medical" if decision.partial_kind == "medical" else "mixed_third_party"
        )
        template = self.boundary.templates[template_name]
        notice = decision.partial_notice or template["summary"]
        if decision.partial_kind == "medical":
            intent_summary = "问题同时包含本公司信息和宠物医疗诉求，只保留可独立回答的本公司部分。"
            evidence_summary = "未提供任何疾病诊断、治疗或用药方案。"
            conclusion_summary = "保守返回医疗边界说明，并引导查询具体的本公司公开信息。"
        else:
            intent_summary = "问题包含自有产品与其他品牌，只保留自有产品部分。"
            evidence_summary = "未展示任何第三方品牌评价或未经验证的比较结论。"
            conclusion_summary = "保守返回服务范围说明，并引导查询具体自有产品。"
        return ChatResponse(
            request_id=request_id,
            status="partial_answer",
            intent=decision.intent,
            evidence_status="not_applicable",
            fallback_kind="scope_only",
            decision=DecisionView(
                reason_code=reason_code,
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title=template["title"],
                summary=notice,
                refusal_reason=(decision.refusal_reason or template["reason"]),
                sections=[
                    AnswerSection(
                        heading="可以继续了解",
                        content=(
                            "本次没有形成可验证的实质公司或产品答案。"
                            "请直接询问本公司的公开业务，或具体自有品牌、产品名称、"
                            "型号和公开功能。"
                        ),
                    )
                ],
            ),
            thinking_summary=[
                ThinkingStep(
                    stage="intent",
                    summary=intent_summary,
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="evidence",
                    summary=evidence_summary,
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="conclusion",
                    summary=conclusion_summary,
                    evidence_ids=[],
                ),
            ],
            sources=[],
            confidence=None,
            follow_up_questions=self._self_only_follow_ups(decision),
            metrics={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "retrieved_chunks": retrieved_chunks,
            },
        )

    def _service_unavailable_response(
        self, request_id: str, decision: BoundaryDecision, *, retrieval_ms: int = 0
    ) -> ChatResponse:
        template = self.boundary.templates["service_unavailable"]
        unavailable = BoundaryDecision(
            mode="fixed",
            status="service_unavailable",
            evidence_status="not_applicable",
            reason_code="LOCAL_MODEL_UNAVAILABLE",
            policy_version=decision.policy_version,
            intent=decision.intent,
            title=template["title"],
            summary=template["summary"],
            refusal_reason=template.get("reason"),
            follow_up_questions=("稍后重试",),
        )
        response = self._fixed_response(request_id, unavailable)
        response.decision.route_reason_code = decision.reason_code
        response.metrics["retrieval_ms"] = retrieval_ms
        return response

    async def _general_knowledge_response(
        self,
        request_id: str,
        request: ChatRequest,
        decision: BoundaryDecision,
        deadline: float | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        generation_started = time.perf_counter()
        initial_generation_ms = 0
        repair_ms = 0
        repair_attempts = 0
        repair_reason = ""
        repair_timed_out = False
        original_salvage: tuple[GeneralKnowledgeAnswer, GuardResult] | None = None
        try:
            async with self._model_stage_budget(deadline, "通用知识生成"):
                generated = await self.ollama.answer_general_knowledge(
                    request.question, self._sanitized_history(request.history)
                )
            initial_generation_ms = round((time.perf_counter() - generation_started) * 1000)
            if progress is not None:
                progress("validating")
            guard = validate_general_knowledge_answer(
                generated, self.boundary.blocked_third_party_aliases
            )
            if not guard.valid:
                if progress is not None:
                    progress("repairing")
                repair_attempts = 1
                repair_reason = guard.reason
                logger.warning(
                    "general answer guard failed request_id=%s reason=%s generation_ms=%d",
                    request_id,
                    guard.reason,
                    round((time.perf_counter() - generation_started) * 1000),
                )
                original_salvage = salvage_general_knowledge_answer(
                    generated, self.boundary.blocked_third_party_aliases
                )
                repair_started = time.perf_counter()
                try:
                    async with self._model_stage_budget(
                        deadline,
                        "通用知识修复",
                        cap_seconds=self.settings.model_repair_budget_seconds,
                    ):
                        repaired = await self.ollama.repair_general_knowledge(
                            request.question,
                            generated,
                            guard.reason,
                        )
                except (ModelStageBudgetExceeded, OllamaTimeoutError):
                    repair_ms = round((time.perf_counter() - repair_started) * 1000)
                    repair_timed_out = True
                    logger.warning(
                        "general answer repair timed out request_id=%s reason=%s "
                        "fallback=%s repair_ms=%d",
                        request_id,
                        guard.reason,
                        "salvaged" if original_salvage is not None else "none",
                        repair_ms,
                    )
                    if original_salvage is None:
                        raise
                    generated, guard = original_salvage
                except RuntimeError as exc:
                    if isinstance(exc, OllamaUnavailableError):
                        raise
                    repair_ms = round((time.perf_counter() - repair_started) * 1000)
                    logger.warning(
                        "general answer repair output invalid request_id=%s reason=%s "
                        "fallback=%s repair_ms=%d",
                        request_id,
                        guard.reason,
                        "salvaged" if original_salvage is not None else "none",
                        repair_ms,
                    )
                    if original_salvage is None:
                        raise
                    generated, guard = original_salvage
                else:
                    repair_ms = round((time.perf_counter() - repair_started) * 1000)
                    generated = repaired
                    guard = validate_general_knowledge_answer(
                        generated, self.boundary.blocked_third_party_aliases
                    )
                    if not guard.valid:
                        logger.warning(
                            "general answer repair failed guard request_id=%s reason=%s "
                            "repair_ms=%d",
                            request_id,
                            guard.reason,
                            repair_ms,
                        )
            if not guard.valid:
                salvaged = salvage_general_knowledge_answer(
                    generated,
                    self.boundary.blocked_third_party_aliases,
                )
                if salvaged is not None:
                    generated, guard = salvaged
                elif original_salvage is not None:
                    generated, guard = original_salvage
        except OllamaTimeoutError:
            raise
        except OllamaUnavailableError:
            return self._service_unavailable_response(request_id, decision)
        except RuntimeError:
            generation_ms = int((time.perf_counter() - generation_started) * 1000)
            response = self._insufficient_response(
                request_id,
                decision,
                "本地模型未能生成符合安全边界的通用养宠回答，请稍后重试。",
                reason_code="GENERAL_MODEL_OUTPUT_INVALID",
                generation_ms=generation_ms,
            )
            response.follow_up_questions = ["换一种更具体的日常养宠问法", "稍后重试"]
            return response

        generation_ms = int((time.perf_counter() - generation_started) * 1000)
        if not guard.valid:
            response = self._insufficient_response(
                request_id,
                decision,
                "通用知识回答未通过安全边界校验，因此没有展示可能误导的内容。",
                reason_code=f"GENERAL_KNOWLEDGE_GUARD_{guard.reason}",
                generation_ms=generation_ms,
            )
            response.follow_up_questions = [
                "换一种更具体的日常养宠问法",
                "若涉及健康异常，请咨询兽医",
            ]
            return response

        return ChatResponse(
            request_id=request_id,
            status="general_knowledge",
            intent=decision.intent,
            evidence_status="model_knowledge",
            decision=DecisionView(
                reason_code=decision.reason_code,
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title=generated.title,
                summary=generated.summary,
                sections=generated.sections,
            ),
            thinking_summary=[
                ThinkingStep(
                    stage="intent",
                    summary="问题属于非诊疗性质的日常养宠通用知识。",
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="evidence",
                    summary="本次只使用本地模型自身通用知识，未检索或引用上传文档。",
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="conclusion",
                    summary="回答已通过诊疗、用药、品牌内容和虚假文档引用检查。",
                    evidence_ids=[],
                ),
            ],
            sources=[],
            confidence=None,
            follow_up_questions=generated.follow_up_questions,
            metrics={
                "retrieval_ms": 0,
                "generation_ms": generation_ms,
                "retrieved_chunks": 0,
                "repair_ms": repair_ms,
                "initial_generation_ms": initial_generation_ms,
                "repair_attempts": repair_attempts,
                "repair_reason": repair_reason,
                "repair_timed_out": int(repair_timed_out),
            },
        )

    async def chat(
        self,
        request: ChatRequest,
        *,
        deadline: float | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        def emit(stage: str) -> None:
            if progress is not None:
                progress(stage)

        request_id = str(uuid.uuid4())
        resolved_question = self.resolve_question(request)
        resolved_request = request.model_copy(update={"question": resolved_question})
        decision = self.boundary.classify(resolved_question)
        if decision.mode == "fixed":
            response = self._fixed_response(request_id, decision)
            self._record(request.question, response)
            return response

        topic_check_ms = 0
        if decision.verify_topic:
            emit("classifying")
            prior_pet_subject: str | None = None
            recent_user_questions: tuple[str, ...] = ()
            if resolved_question == request.question and not self.boundary.has_explicit_pet_subject(
                request.question
            ):
                pet_context = self._recent_pet_context(request.history)
                state = self._history_state(pet_context)
                if not state.ambiguous and state.topic != "product":
                    recent_user_questions = tuple(
                        message.content for message in pet_context if message.role == "user"
                    )
                    if state.topic == "pet_care" and state.subject:
                        prior_pet_subject = state.subject
            topic_check_started = time.perf_counter()
            try:
                async with self._model_stage_budget(
                    deadline,
                    "问题主题判别",
                    cap_seconds=self.settings.topic_classification_timeout_seconds,
                ):
                    classification = await self.ollama.classify_question_topic(
                        resolved_question,
                        self.boundary.company_name,
                        self.boundary.owned_aliases,
                        prior_pet_subject,
                        recent_user_questions,
                    )
                topic_kind = classification.kind
                if (
                    topic_kind == "pet_care"
                    and classification.use_prior_pet_subject
                    and not self.boundary.has_explicit_pet_subject(request.question)
                ):
                    subject = classification.referenced_pet_subject or prior_pet_subject
                    if subject and any(subject in item for item in recent_user_questions):
                        resolved_question = f"{subject}{resolved_question}"
                        resolved_request = request.model_copy(
                            update={"question": resolved_question}
                        )
                    else:
                        topic_kind = "uncertain"
            except OllamaTimeoutError:
                raise
            except OllamaUnavailableError:
                response = self._service_unavailable_response(request_id, decision)
                self._record(request.question, response)
                return response
            except RuntimeError:
                logger.warning("question topic classification invalid request_id=%s", request_id)
                topic_kind = "uncertain"
            topic_check_ms = round((time.perf_counter() - topic_check_started) * 1000)
            decision = self.boundary.resolve_topic(decision, topic_kind)
            if decision.mode == "fixed":
                response = self._fixed_response(request_id, decision)
                response.metrics["topic_check_ms"] = topic_check_ms
                self._record(request.question, response)
                return response

        if decision.mode == "general_model":
            emit("generating")
            response = await self._general_knowledge_response(
                request_id, resolved_request, decision, deadline, progress
            )
            response.metrics["topic_check_ms"] = topic_check_ms
            self._record(request.question, response)
            return response

        safe_question = self._self_only_question(resolved_question, decision)
        safe_request = resolved_request.model_copy(update={"question": safe_question})
        content_mode = self._response_content_mode(request.question)
        cache_key = self._public_cache_key(request, decision, safe_question, content_mode)
        if cache_key is not None:
            cached = self.answer_cache.get(cache_key)
            if cached is not None:
                if self._public_cache_key(
                    request, decision, safe_question, content_mode
                ) == cache_key and self._cached_sources_active(cached):
                    cached.request_id = request_id
                    cached.metrics.update(
                        {
                            "retrieval_ms": 0,
                            "generation_ms": 0,
                            "retrieved_chunks": 0,
                            "initial_generation_ms": 0,
                            "repair_ms": 0,
                            "repair_attempts": 0,
                            "repair_reason": "",
                            "cache_hit": 1,
                            "topic_check_ms": topic_check_ms,
                        }
                    )
                    self._record(request.question, cached)
                    return cached
                self.answer_cache.discard(cache_key)

        emit("retrieving")

        if content_mode == "image_only":
            media_started = time.perf_counter()
            images = self._select_response_images(safe_request, [])
            media_ms = int((time.perf_counter() - media_started) * 1000)
            if images:
                if not self._media_snapshots_are_active(images):
                    response = self._insufficient_response(
                        request_id,
                        decision,
                        "图片来源在回答完成前已被撤回或替换，请重新查询。",
                        reason_code="SOURCE_CHANGED_DURING_RESPONSE",
                        retrieval_ms=media_ms,
                    )
                    self._record(request.question, response)
                    return response
                response = self._image_response(
                    request_id,
                    request,
                    decision,
                    [],
                    images,
                    media_ms,
                )
                self._record(request.question, response)
                return response

        chunks = self.repository.fetch_searchable_chunks(decision.domains)
        if not chunks:
            if content_mode == "image_only":
                response = self._insufficient_response(
                    request_id,
                    decision,
                    "没有找到与当前品牌、品类、型号和款式匹配的已审核产品图片。",
                    reason_code="PRODUCT_MEDIA_NOT_FOUND",
                )
                response.follow_up_questions = ["请提供更具体的产品名称或型号"]
                self._record(request.question, response)
                return response
            response = self._insufficient_response(
                request_id,
                decision,
                "知识库尚无与该问题匹配的已审核发布资料。",
                reason_code="NO_PUBLISHED_EVIDENCE",
            )
            self._record(request.question, response)
            return response

        retrieval_started = time.perf_counter()
        try:
            query_embedding = (await self.ollama.embed([safe_question]))[0]
        except OllamaTimeoutError:
            raise
        except OllamaUnavailableError:
            retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
            response = self._service_unavailable_response(
                request_id, decision, retrieval_ms=retrieval_ms
            )
            self._record(request.question, response)
            return response
        results = retrieve(
            safe_question,
            query_embedding,
            chunks,
            top_k=self.settings.retrieval_top_k,
        )
        results = self._ensure_requested_model_coverage(
            safe_question,
            query_embedding,
            chunks,
            results,
            top_k=self.settings.retrieval_top_k,
        )
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
        if content_mode == "image_only":
            images = self._select_response_images(safe_request, results)
            if images:
                if not self._media_snapshots_are_active(images):
                    response = self._insufficient_response(
                        request_id,
                        decision,
                        "图片来源在回答完成前已被撤回或替换，请重新查询。",
                        reason_code="SOURCE_CHANGED_DURING_RESPONSE",
                        retrieval_ms=retrieval_ms,
                        retrieved_chunks=len(results),
                    )
                    self._record(request.question, response)
                    return response
                response = self._image_response(
                    request_id,
                    request,
                    decision,
                    results,
                    images,
                    retrieval_ms,
                )
                self._record(request.question, response)
                return response
            response = self._insufficient_response(
                request_id,
                decision,
                "已检索到相关文字资料，但没有与当前问题匹配且已审核发布的产品图片。",
                reason_code="PRODUCT_MEDIA_NOT_FOUND",
                retrieval_ms=retrieval_ms,
                retrieved_chunks=len(results),
            )
            response.follow_up_questions = ["请提供更具体的产品名称或型号"]
            self._record(request.question, response)
            return response
        if not results or results[0].source.score < self.settings.min_retrieval_score:
            response = self._insufficient_response(
                request_id,
                decision,
                "现有已发布资料与这个问题的相关度不足，暂时无法可靠回答。",
                reason_code="LOW_RETRIEVAL_RELEVANCE",
                retrieval_ms=retrieval_ms,
                retrieved_chunks=len(results),
            )
            self._record(request.question, response)
            return response

        if self._missing_strict_evidence(safe_question, results):
            response = self._insufficient_response(
                request_id,
                decision,
                "现有已审核发布资料未包含当前问题所需的直接信息，暂时无法可靠确认。",
                reason_code="NO_DIRECT_EVIDENCE_FOR_REQUESTED_FACT",
                retrieval_ms=retrieval_ms,
                retrieved_chunks=len(results),
            )
            response.follow_up_questions = ["请补充对应字段的最新已审核资料"]
            self._record(request.question, response)
            return response

        if self._has_capacity_conflict(safe_question, results):
            response = self._insufficient_response(
                request_id,
                decision,
                "当前已审核资料对同一产品的容量记载不一致，暂时无法确认唯一数值；"
                "请先核实适用型号、版本和最新生效资料。",
                reason_code="CONFLICTING_PUBLISHED_EVIDENCE",
                retrieval_ms=retrieval_ms,
                retrieved_chunks=len(results),
            )
            response.follow_up_questions = ["请核实具体产品版本和最新参数表"]
            self._record(request.question, response)
            return response

        context, used_results = build_context(results)
        if not used_results:
            response = self._insufficient_response(
                request_id,
                decision,
                "候选资料未能放入本次证据上下文，暂时无法可靠回答。",
                reason_code="EMPTY_MODEL_CONTEXT",
                retrieval_ms=retrieval_ms,
                retrieved_chunks=len(results),
            )
            self._record(request.question, response)
            return response

        allowed_source_ids = [result.source.source_id for result in used_results]
        generation_question = self._generation_question(safe_question)
        salvaged_claims = 0
        extractive_fallback_used = False
        initial_generation_ms = 0
        repair_ms = 0
        repair_attempts = 0
        repair_reason = ""
        repair_timed_out = False
        answer_timed_out = False
        generation_started = time.perf_counter()
        emit("generating")
        try:
            try:
                async with self._model_stage_budget(deadline, "初稿生成"):
                    generated = await self.ollama.answer(
                        generation_question,
                        context,
                        [] if decision.partial_notice else self._sanitized_history(request.history),
                        allowed_source_ids,
                    )
                initial_generation_ms = round((time.perf_counter() - generation_started) * 1000)
                emit("validating")
            except (ModelStageBudgetExceeded, OllamaTimeoutError):
                answer_timed_out = True
                raise
            guard = validate_generated_answer(
                generated,
                used_results,
                self.boundary.blocked_third_party_aliases,
            )
            if not guard.valid:
                emit("repairing")
                repair_attempts = 1
                repair_reason = guard.reason
                logger.warning(
                    "grounded answer guard failed request_id=%s reason=%s generation_ms=%d",
                    request_id,
                    guard.reason,
                    round((time.perf_counter() - generation_started) * 1000),
                )
                original_salvage = salvage_supported_answer(
                    generated,
                    used_results,
                    self.boundary.blocked_third_party_aliases,
                )
                if original_salvage is not None and (
                    original_salvage[0].status != "grounded"
                    or not self._salvage_covers_requested_facts(safe_question, original_salvage[0])
                ):
                    original_salvage = None
                repair_started = time.perf_counter()
                try:
                    async with self._model_stage_budget(
                        deadline,
                        "证据修复",
                        cap_seconds=self.settings.model_repair_budget_seconds,
                    ):
                        repaired = await self.ollama.repair_answer(
                            generation_question,
                            context,
                            generated,
                            allowed_source_ids,
                            guard.reason,
                            guard.repair_detail,
                        )
                except (ModelStageBudgetExceeded, OllamaTimeoutError):
                    repair_ms = round((time.perf_counter() - repair_started) * 1000)
                    repair_timed_out = True
                    logger.warning(
                        "grounded answer repair timed out request_id=%s reason=%s "
                        "fallback=%s repair_ms=%d",
                        request_id,
                        guard.reason,
                        "salvaged" if original_salvage is not None else "pending_extractive",
                        repair_ms,
                    )
                    if original_salvage is None:
                        raise
                    generated, guard, salvaged_claims = original_salvage
                else:
                    repair_ms = round((time.perf_counter() - repair_started) * 1000)
                    generated = repaired
                    guard = validate_generated_answer(
                        generated,
                        used_results,
                        self.boundary.blocked_third_party_aliases,
                    )
                    if not guard.valid:
                        logger.warning(
                            "grounded answer repair failed guard request_id=%s reason=%s "
                            "repair_ms=%d",
                            request_id,
                            guard.reason,
                            repair_ms,
                        )
            if not guard.valid:
                salvaged = salvage_supported_answer(
                    generated,
                    used_results,
                    self.boundary.blocked_third_party_aliases,
                )
                if salvaged is not None:
                    generated, guard, salvaged_claims = salvaged
            if not guard.valid or generated.status == "insufficient_evidence":
                fallback = self._extractive_fallback(
                    safe_question,
                    decision,
                    used_results,
                    guard.reason,
                )
                if fallback is not None:
                    generated, guard = fallback
                    extractive_fallback_used = True
        except (ModelStageBudgetExceeded, OllamaTimeoutError):
            fallback = self._extractive_fallback(
                safe_question,
                decision,
                used_results,
                "MODEL_OUTPUT_INVALID",
            )
            if fallback is None:
                raise
            generated, guard = fallback
            extractive_fallback_used = True
            logger.warning(
                "grounded answer model stage timed out request_id=%s "
                "fallback=verified_extractive answer_timed_out=%s repair_timed_out=%s",
                request_id,
                answer_timed_out,
                repair_timed_out,
            )
        except OllamaUnavailableError:
            response = self._service_unavailable_response(
                request_id, decision, retrieval_ms=retrieval_ms
            )
            self._record(request.question, response)
            return response
        except RuntimeError:
            generation_ms = int((time.perf_counter() - generation_started) * 1000)
            fallback = self._extractive_fallback(
                safe_question,
                decision,
                used_results,
                "MODEL_OUTPUT_INVALID",
            )
            if fallback is not None:
                generated, guard = fallback
                extractive_fallback_used = True
            else:
                if decision.partial_notice:
                    response = self._self_only_safe_fallback(
                        request_id,
                        decision,
                        reason_code="SELF_ONLY_SAFE_FALLBACK_MODEL_OUTPUT_INVALID",
                        retrieval_ms=retrieval_ms,
                        generation_ms=generation_ms,
                        retrieved_chunks=len(used_results),
                    )
                    self._record(request.question, response)
                    return response
                response = self._insufficient_response(
                    request_id,
                    decision,
                    "已有资料，但本次生成未能形成符合结构和证据要求的答案，请重新尝试。",
                    reason_code="MODEL_OUTPUT_INVALID",
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    retrieved_chunks=len(used_results),
                )
                response.follow_up_questions = [request.question]
                self._record(request.question, response)
                return response
        generation_ms = int((time.perf_counter() - generation_started) * 1000)

        if not guard.valid:
            if decision.partial_notice:
                response = self._self_only_safe_fallback(
                    request_id,
                    decision,
                    reason_code=f"SELF_ONLY_SAFE_FALLBACK_{guard.reason}",
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    retrieved_chunks=len(used_results),
                )
                self._record(request.question, response)
                return response
            response = self._insufficient_response(
                request_id,
                decision,
                "已找到相关资料，但本次生成未能稳定绑定到对应来源，请重新尝试。",
                reason_code=f"EVIDENCE_GUARD_{guard.reason}",
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                retrieved_chunks=len(used_results),
            )
            response.follow_up_questions = [request.question]
            self._record(request.question, response)
            return response

        if generated.status == "insufficient_evidence":
            if decision.partial_notice:
                response = self._self_only_safe_fallback(
                    request_id,
                    decision,
                    reason_code="SELF_ONLY_SAFE_FALLBACK_NO_EVIDENCE",
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    retrieved_chunks=len(used_results),
                )
                self._record(request.question, response)
                return response
            response = self._insufficient_response(
                request_id,
                decision,
                generated.summary,
                reason_code="MODEL_REPORTED_EVIDENCE_GAP",
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                retrieved_chunks=len(used_results),
            )
            self._record(request.question, response)
            return response

        cited_source_ids = set(guard.cited_source_ids)
        if not self._source_snapshots_are_active(used_results, cited_source_ids):
            response = self._insufficient_response(
                request_id,
                decision,
                "回答生成期间对应资料已被撤回或发布了新版本，请重新查询。",
                reason_code="SOURCE_CHANGED_DURING_GENERATION",
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                retrieved_chunks=len(used_results),
            )
            response.follow_up_questions = [request.question]
            self._record(request.question, response)
            return response

        source_lookup = {result.source.source_id: result.source for result in used_results}
        excerpt_lookup = dict(guard.source_excerpts)
        sources = [
            source_lookup[source_id].model_copy(
                update={"excerpt": excerpt_lookup.get(source_id, source_lookup[source_id].excerpt)}
            )
            for source_id in guard.cited_source_ids
        ]
        cited_results = [
            result for result in used_results if result.source.source_id in guard.cited_source_ids
        ]
        generated_summary = render_validated_citations(
            generated.summary,
            generated.summary_source_ids,
        )
        sections = [
            AnswerSection(
                heading=section.heading,
                content=render_validated_citations(section.content, section.source_ids),
            )
            for section in generated.sections
        ]
        summary = generated_summary
        follow_up_questions = generated.follow_up_questions
        if decision.partial_notice:
            summary = decision.partial_notice
            sections.insert(
                0,
                AnswerSection(heading="自有产品信息", content=generated_summary),
            )
            follow_up_questions = self._self_only_follow_ups(decision)

        response_status = decision.status
        evidence_status = decision.evidence_status
        reason_code = decision.reason_code

        response_images: list[ResponseImage] = []
        if content_mode == "text_and_image":
            candidate_images = self._select_response_images(
                safe_request,
                cited_results,
                limit=4,
            )
            if candidate_images and self._media_snapshots_are_active(candidate_images):
                response_images = candidate_images
                sources.extend(self._media_source_refs(response_images))

        evidence_ids = [source.source_id for source in sources]
        document_names = list(dict.fromkeys(source.document_title for source in sources))
        conclusion = (
            "只回答服务范围内且有资料支持的部分。"
            if response_status == "partial_answer"
            else "引用校验通过，答案仅使用本次实际提供给模型的证据。"
        )
        if extractive_fallback_used:
            conclusion = (
                "生成结果未通过证据校验，已改用已审核来源中的原文摘录，并再次通过逐条来源校验。"
            )
        elif salvaged_claims:
            conclusion = (
                f"已修正 {salvaged_claims} 处未通过证据校验的断言或冗余来源，"
                "仅保留逐条验证通过的内容。"
            )
        response = ChatResponse(
            request_id=request_id,
            status=response_status,
            intent=decision.intent,
            evidence_status=evidence_status,
            fallback_kind=("verified_extractive" if extractive_fallback_used else None),
            decision=DecisionView(
                reason_code=reason_code,
                policy_version=decision.policy_version,
                route_reason_code=decision.reason_code,
            ),
            answer=AnswerPayload(
                title=generated.title,
                summary=summary,
                refusal_reason=decision.refusal_reason,
                sections=sections,
            ),
            images=response_images,
            thinking_summary=[
                ThinkingStep(
                    stage="intent",
                    summary="问题通过服务范围检查，进入已发布知识检索。",
                    evidence_ids=[],
                ),
                ThinkingStep(
                    stage="retrieval",
                    summary=(
                        f"在 {len(chunks)} 个可检索片段中进行混合检索，"
                        f"实际向模型提供 {len(used_results)} 个片段。"
                    ),
                    evidence_ids=evidence_ids,
                ),
                ThinkingStep(
                    stage="evidence",
                    summary=f"答案引用并校验了：{'、'.join(document_names)}。",
                    evidence_ids=evidence_ids,
                ),
                ThinkingStep(stage="conclusion", summary=conclusion, evidence_ids=evidence_ids),
            ],
            sources=sources,
            confidence=None,
            follow_up_questions=follow_up_questions,
            metrics={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "retrieved_chunks": len(used_results),
                "topic_check_ms": topic_check_ms,
                "salvaged_claims": salvaged_claims,
                "extractive_fallback": int(extractive_fallback_used),
                "repair_ms": repair_ms,
                "initial_generation_ms": initial_generation_ms,
                "repair_attempts": repair_attempts,
                "repair_reason": repair_reason,
                "repair_timed_out": int(repair_timed_out),
                "answer_timed_out": int(answer_timed_out),
            },
        )
        if (
            cache_key is not None
            and response.status == "grounded"
            and response.evidence_status == "supported"
            and response.fallback_kind is None
            and self._public_cache_key(request, decision, safe_question, content_mode) == cache_key
            and self._cached_sources_active(response)
        ):
            self.answer_cache.put(cache_key, response)
        self._record(request.question, response)
        return response
