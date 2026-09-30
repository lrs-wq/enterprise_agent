import asyncio
import hashlib
import ipaddress
import json
import logging
import secrets
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, TypeVar

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from company_agent.config import PROJECT_ROOT, Settings, get_settings
from company_agent.database import KnowledgeRepository
from company_agent.document_parser import SUPPORTED_EXTENSIONS
from company_agent.ollama_client import (
    OllamaClient,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from company_agent.runtime_control import (
    CapacityAdmission,
    CapacityGate,
    ClientDisconnectedError,
    ServiceOverloadedError,
    SharedCapacityGate,
    SlidingWindowLimiter,
    run_until_disconnected,
)
from company_agent.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkUpdate,
    DocumentUpdate,
    MediaAssetUpdate,
)
from company_agent.services import (
    AgentService,
    KnowledgeService,
    ModelStageBudgetExceeded,
    safe_filename,
)

logger = logging.getLogger(__name__)

settings = get_settings()
repository = KnowledgeRepository(settings.database_path)


def make_capacity_gate(
    resource: str,
    *,
    max_active: int,
    max_queue: int,
    retry_after_seconds: int = 5,
) -> CapacityGate:
    if settings.shared_capacity_enabled:
        return SharedCapacityGate(
            resource,
            database_path=settings.capacity_database_path,
            max_active=max_active,
            max_queue=max_queue,
            retry_after_seconds=retry_after_seconds,
        )
    return CapacityGate(
        resource,
        max_active=max_active,
        max_queue=max_queue,
        retry_after_seconds=retry_after_seconds,
    )


chat_gate = make_capacity_gate(
    "聊天请求",
    max_active=settings.chat_max_concurrency,
    max_queue=settings.chat_max_queue,
)
model_gate = make_capacity_gate(
    "本地模型",
    max_active=settings.model_max_concurrency,
    max_queue=settings.model_max_queue,
)
admin_task_gate = make_capacity_gate(
    "后台处理任务",
    max_active=settings.admin_task_max_concurrency,
    max_queue=settings.admin_task_max_queue,
    retry_after_seconds=10,
)
upload_ingress_gate = make_capacity_gate(
    "上传接收",
    max_active=1,
    max_queue=1,
    retry_after_seconds=10,
)
chat_rate_limiter = SlidingWindowLimiter(
    limit=settings.chat_rate_limit_requests,
    window_seconds=settings.chat_rate_limit_window_seconds,
)
admin_auth_limiter = SlidingWindowLimiter(
    limit=settings.admin_auth_failure_limit,
    window_seconds=settings.admin_auth_failure_window_seconds,
)
ollama = OllamaClient(
    base_url=settings.ollama_base_url,
    chat_model=settings.chat_model,
    embedding_model=settings.embedding_model,
    model_context=settings.model_context,
    capacity_gate=model_gate,
    model_timeout_seconds=settings.model_request_timeout_seconds,
    embedding_timeout_seconds=settings.embedding_request_timeout_seconds,
)
knowledge_service = KnowledgeService(settings, repository, ollama)
agent_service = AgentService(settings, repository, ollama)
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    repository.initialize()
    await asyncio.to_thread(knowledge_service.materialize_active_media_snapshots)
    try:
        yield
    finally:
        await ollama.aclose()


app = FastAPI(
    title="Company AI Agent",
    version="0.1.0",
    description="局域网本地模型公司知识助手",
    lifespan=lifespan,
)


class RequestBodyTooLargeError(RuntimeError):
    pass


class RequestBodyLimitMiddleware:
    """Enforce body limits while bytes arrive, including chunked direct requests."""

    def __init__(self, app: ASGIApp, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limits.get(str(scope.get("path", "")))
        if limit is None:
            await self.app(scope, receive, send)
            return
        request_headers = Headers(raw=list(scope.get("headers", [])))
        raw_content_length = request_headers.get("content-length")
        if raw_content_length:
            try:
                content_length = int(raw_content_length)
            except ValueError:
                response = _problem_response(
                    400,
                    "INVALID_CONTENT_LENGTH",
                    "请求体长度格式无效",
                )
                await response(scope, receive, send)
                return
            if content_length > limit:
                response = _problem_response(
                    413,
                    "REQUEST_BODY_TOO_LARGE",
                    "请求体超过服务允许的大小",
                )
                await response(scope, receive, send)
                return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestBodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise
            response = _problem_response(
                413,
                "REQUEST_BODY_TOO_LARGE",
                "请求体超过服务允许的大小",
            )
            await response(scope, receive, send)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        prefix = headers.get("x-forwarded-prefix", "").rstrip("/")
        allowed_prefix = settings.public_base_path.rstrip("/")
        state = scope.setdefault("state", {})
        state["base_path"] = prefix if prefix and prefix == allowed_prefix else ""
        if scope.get("method") == "POST" and scope.get("path") == "/api/admin/documents":
            deadline = asyncio.get_running_loop().time() + settings.admin_task_timeout_seconds
            try:
                async with asyncio.timeout_at(deadline):
                    async with upload_ingress_gate.slot(deadline=deadline):
                        await self.app(scope, receive, send)
            except ServiceOverloadedError as exc:
                response = _problem_response(
                    429,
                    "UPLOAD_INGRESS_BUSY",
                    "当前已有上传正在接收，请稍后重试。",
                    retry_after=exc.retry_after_seconds,
                )
                await response(scope, receive, send)
            except TimeoutError:
                response = _problem_response(
                    504,
                    "UPLOAD_DEADLINE_EXCEEDED",
                    "上传或处理超过时限，已停止继续等待。",
                    retry_after=10,
                )
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(
    RequestBodyLimitMiddleware,
    limits={
        "/api/chat": settings.max_chat_request_kb * 1024,
        "/api/chat/stream": settings.max_chat_request_kb * 1024,
        "/api/admin/documents": settings.max_upload_request_mb * 1024 * 1024,
    },
)
app.add_middleware(RequestContextMiddleware)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")


def _problem_response(
    status_code: int,
    code: str,
    message: str,
    *,
    retry_after: int | None = None,
) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": {
                "code": code,
                "message": message,
                "retryable": retry_after is not None,
            }
        },
        headers=headers,
    )


def _problem_exception(
    status_code: int,
    code: str,
    message: str,
    *,
    retry_after: int | None = None,
) -> HTTPException:
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "retryable": retry_after is not None},
        headers=headers,
    )


def _client_key(request: Request) -> str:
    peer = request.client.host if request.client is not None else "unknown"
    candidate = peer
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        peer_address = None
    if peer_address is not None and peer_address.is_loopback:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        try:
            candidate = str(ipaddress.ip_address(forwarded)) if forwarded else peer
        except ValueError:
            candidate = peer
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:24]


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


AdminResultT = TypeVar("AdminResultT")


async def _run_admin_task(
    request: Request,
    operation: Callable[[], Awaitable[AdminResultT]],
) -> tuple[AdminResultT, CapacityAdmission]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.admin_task_timeout_seconds

    async def execute() -> tuple[AdminResultT, CapacityAdmission]:
        async with admin_task_gate.slot(deadline=deadline) as admission:
            return await operation(), admission

    try:
        async with asyncio.timeout_at(deadline):
            return await run_until_disconnected(
                execute(),
                lambda: _wait_for_disconnect(request),
            )
    except ServiceOverloadedError as exc:
        raise _problem_exception(
            429,
            "ADMIN_TASK_BUSY",
            "后台已有处理任务，请稍后重试。",
            retry_after=exc.retry_after_seconds,
        ) from exc
    except TimeoutError as exc:
        raise _problem_exception(
            504,
            "ADMIN_TASK_DEADLINE_EXCEEDED",
            "后台任务超过处理时限，已停止继续等待。",
            retry_after=10,
        ) from exc
    except ClientDisconnectedError as exc:
        raise _problem_exception(499, "CLIENT_DISCONNECTED", "客户端已取消请求。") from exc
    except OllamaTimeoutError as exc:
        raise _problem_exception(
            504,
            "MODEL_TIMEOUT",
            str(exc),
            retry_after=10,
        ) from exc
    except OllamaUnavailableError as exc:
        raise _problem_exception(
            503,
            "MODEL_UNAVAILABLE",
            str(exc),
            retry_after=10,
        ) from exc


async def _copy_upload_limited(upload: UploadFile, destination: Path) -> int:
    maximum = settings.max_file_mb * 1024 * 1024
    size = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ValueError(f"文件超过 {settings.max_file_mb}MB 限制")
            await asyncio.to_thread(handle.write, chunk)
    return size


def _set_admission_header(response: Response, admission: CapacityAdmission) -> None:
    response.headers["X-AI-Queue-State"] = "queued" if admission.queued else "admitted"


def media_file_response(asset: dict[str, object]) -> FileResponse:
    path = Path(str(asset["stored_path"])).resolve()
    allowed_roots = (settings.media_dir.resolve(), settings.published_media_dir.resolve())
    if not any(path.is_relative_to(root) for root in allowed_roots) or not path.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    return FileResponse(
        path,
        media_type=str(asset["mime_type"]),
        headers={"Cache-Control": "private, max-age=3600"},
    )


async def require_admin(
    request: Request,
    configured: Annotated[Settings, Depends(get_settings)],
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    if not configured.admin_auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理员令牌尚未配置，后台接口已安全锁定",
        )
    client_key = _client_key(request)
    retry_after = admin_auth_limiter.retry_after(client_key)
    if retry_after is not None:
        raise _problem_exception(
            429,
            "ADMIN_AUTH_RATE_LIMITED",
            "管理员令牌连续验证失败次数过多，请稍后再试。",
            retry_after=retry_after,
        )
    supplied = x_admin_token or ""
    matches_current = secrets.compare_digest(supplied, configured.admin_token)
    matches_previous = configured.admin_previous_token_enabled and secrets.compare_digest(
        supplied,
        configured.admin_previous_token,
    )
    if not matches_current and not matches_previous:
        admin_auth_limiter.record(client_key)
        retry_after = admin_auth_limiter.retry_after(client_key)
        if retry_after is not None:
            raise _problem_exception(
                429,
                "ADMIN_AUTH_RATE_LIMITED",
                "管理员令牌连续验证失败次数过多，请稍后再试。",
                retry_after=retry_after,
            )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="管理员令牌无效")
    admin_auth_limiter.reset(client_key)


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"base_path": request.state.base_path},
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={"base_path": request.state.base_path},
    )


@app.get("/admin/documents/{document_id}", response_class=HTMLResponse)
async def document_detail_page(request: Request, document_id: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="document_detail.html",
        context={
            "document_id": document_id,
            "base_path": request.state.base_path,
        },
    )


@app.get("/api/health")
async def health() -> dict[str, object]:
    try:
        models = await ollama.list_models()
        ollama_status = "ok"
    except OllamaUnavailableError:
        models = []
        ollama_status = "unavailable"
    chat_model_ready = any(name.startswith(settings.chat_model) for name in models)
    embedding_model_ready = any(name.startswith(settings.embedding_model) for name in models)
    fully_ready = ollama_status == "ok" and chat_model_ready and embedding_model_ready
    return {
        "status": "ok" if fully_ready else "degraded",
        "ollama": ollama_status,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "chat_model_ready": chat_model_ready,
        "embedding_model_ready": embedding_model_ready,
        "admin_auth_enabled": settings.admin_auth_enabled,
        "knowledge": repository.stats(),
        "capacity": {
            "chat": asdict(chat_gate.snapshot()),
            "model": asdict(model_gate.snapshot()),
            "admin_tasks": asdict(admin_task_gate.snapshot()),
            "upload_ingress": asdict(upload_ingress_gate.snapshot()),
        },
        "request_limits": {
            "chat_timeout_seconds": settings.chat_request_timeout_seconds,
            "chat_rate_limit_requests": settings.chat_rate_limit_requests,
            "chat_rate_limit_window_seconds": settings.chat_rate_limit_window_seconds,
        },
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: Request, response: Response, payload: ChatRequest) -> ChatResponse:
    rate = chat_rate_limiter.consume(_client_key(request))
    if not rate.allowed:
        raise _problem_exception(
            429,
            "CHAT_RATE_LIMITED",
            "请求过于频繁，请稍后再试。",
            retry_after=rate.retry_after_seconds,
        )
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + settings.chat_request_timeout_seconds

    async def execute() -> tuple[ChatResponse, CapacityAdmission]:
        async with chat_gate.slot(deadline=deadline) as admission:
            return await agent_service.chat(payload, deadline=deadline), admission

    try:
        async with asyncio.timeout_at(deadline):
            result, admission = await run_until_disconnected(
                execute(),
                lambda: _wait_for_disconnect(request),
            )
    except ServiceOverloadedError as exc:
        raise _problem_exception(
            429,
            "CHAT_BUSY",
            "当前问答请求较多，队列已满，请稍后重试。",
            retry_after=exc.retry_after_seconds,
        ) from exc
    except ModelStageBudgetExceeded as exc:
        logger.warning(
            "chat model stage budget exceeded stage=%s elapsed_ms=%d question_hash=%s",
            exc.stage,
            round((loop.time() - started) * 1000),
            hashlib.sha256(payload.question.encode("utf-8")).hexdigest()[:12],
        )
        raise _problem_exception(
            504,
            "MODEL_STAGE_TIMEOUT",
            "本次生成未能在处理时限内完成，请稍后重试。",
            retry_after=5,
        ) from exc
    except TimeoutError as exc:
        logger.warning(
            "chat request deadline exceeded elapsed_ms=%d question_hash=%s",
            round((loop.time() - started) * 1000),
            hashlib.sha256(payload.question.encode("utf-8")).hexdigest()[:12],
        )
        raise _problem_exception(
            504,
            "CHAT_DEADLINE_EXCEEDED",
            "本次问答超过处理时限，请稍后重试。",
            retry_after=5,
        ) from exc
    except ClientDisconnectedError as exc:
        raise _problem_exception(499, "CLIENT_DISCONNECTED", "客户端已取消请求。") from exc
    except OllamaTimeoutError as exc:
        raise _problem_exception(
            504,
            "MODEL_TIMEOUT",
            str(exc),
            retry_after=10,
        ) from exc
    except OllamaUnavailableError as exc:
        raise _problem_exception(
            503,
            "MODEL_UNAVAILABLE",
            str(exc),
            retry_after=10,
        ) from exc
    except RuntimeError as exc:
        raise _problem_exception(502, "CHAT_PROCESSING_FAILED", str(exc)) from exc
    total_ms = max(0, round((loop.time() - started) * 1000))
    result.metrics["queue_ms"] = admission.wait_ms
    result.metrics["total_ms"] = total_ms
    repository.update_query_runtime(
        result.request_id, queue_ms=admission.wait_ms, total_ms=total_ms
    )
    _set_admission_header(response, admission)
    response.headers["Server-Timing"] = f"queue;dur={admission.wait_ms}, total;dur={total_ms}"
    return result


_PROGRESS_LABELS = {
    "accepted": "已接收请求",
    "queued": "当前请求正在排队",
    "classifying": "正在判断问题主题",
    "retrieving": "正在检索已审核资料",
    "generating": "正在生成回答",
    "validating": "正在核验回答与来源",
    "repairing": "正在修正未通过核验的内容",
}


def _sse_event(name: str, value: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(value, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    rate = chat_rate_limiter.consume(_client_key(request))
    if not rate.allowed:
        raise _problem_exception(
            429,
            "CHAT_RATE_LIMITED",
            "请求过于频繁，请稍后再试。",
            retry_after=rate.retry_after_seconds,
        )
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + settings.chat_request_timeout_seconds

    async def events() -> AsyncIterator[str]:
        progress_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=16)

        def emit(stage: str) -> None:
            if stage in _PROGRESS_LABELS and not progress_queue.full():
                progress_queue.put_nowait(stage)

        async def execute() -> tuple[ChatResponse, CapacityAdmission]:
            async with chat_gate.slot(
                deadline=deadline, on_queued=lambda: emit("queued")
            ) as admission:
                result = await agent_service.chat(payload, deadline=deadline, progress=emit)
                return result, admission

        yield _sse_event("progress", {"stage": "accepted", "message": _PROGRESS_LABELS["accepted"]})
        task = asyncio.create_task(execute())
        try:
            async with asyncio.timeout_at(deadline):
                while not task.done():
                    next_progress = asyncio.create_task(progress_queue.get())
                    try:
                        done, _ = await asyncio.wait(
                            {task, next_progress},
                            timeout=10,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if next_progress in done:
                            stage = next_progress.result()
                            yield _sse_event(
                                "progress", {"stage": stage, "message": _PROGRESS_LABELS[stage]}
                            )
                        elif task not in done:
                            yield ": keep-alive\n\n"
                    finally:
                        if not next_progress.done():
                            next_progress.cancel()
                            with suppress(asyncio.CancelledError):
                                await next_progress
                while not progress_queue.empty():
                    stage = progress_queue.get_nowait()
                    yield _sse_event(
                        "progress", {"stage": stage, "message": _PROGRESS_LABELS[stage]}
                    )
                result, admission = await task
                total_ms = max(0, round((loop.time() - started) * 1000))
                result.metrics["queue_ms"] = admission.wait_ms
                result.metrics["total_ms"] = total_ms
                repository.update_query_runtime(
                    result.request_id, queue_ms=admission.wait_ms, total_ms=total_ms
                )
                yield _sse_event("result", result.model_dump(mode="json"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            elapsed = round((loop.time() - started) * 1000)
            if isinstance(exc, ServiceOverloadedError):
                code, message, status_code = "CHAT_BUSY", "当前问答请求较多，请稍后重试。", 429
            elif isinstance(exc, ModelStageBudgetExceeded):
                code, message, status_code = (
                    "MODEL_STAGE_TIMEOUT",
                    "本次生成未能在处理时限内完成，请稍后重试。",
                    504,
                )
            elif isinstance(exc, TimeoutError):
                code, message, status_code = (
                    "CHAT_DEADLINE_EXCEEDED",
                    "本次问答超过处理时限，请稍后重试。",
                    504,
                )
            elif isinstance(exc, OllamaTimeoutError):
                code, message, status_code = "MODEL_TIMEOUT", str(exc), 504
            elif isinstance(exc, OllamaUnavailableError):
                code, message, status_code = "MODEL_UNAVAILABLE", str(exc), 503
            else:
                logger.exception("streamed chat failed elapsed_ms=%d", elapsed)
                code, message, status_code = (
                    "CHAT_PROCESSING_FAILED",
                    "本次请求未能完成，请稍后重试。",
                    502,
                )
            yield _sse_event("error", {"code": code, "message": message, "status": status_code})
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/media/{asset_id}")
async def get_public_media(asset_id: str) -> FileResponse:
    asset = repository.get_public_media_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="图片不存在或尚未公开")
    return media_file_response(asset)


@app.get("/api/admin/documents", dependencies=[Depends(require_admin)])
async def list_documents() -> dict[str, object]:
    return {"items": repository.list_documents(), "stats": repository.stats()}


@app.get("/api/admin/documents/{document_id}", dependencies=[Depends(require_admin)])
async def get_document(document_id: str) -> dict[str, object]:
    document = repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document": document,
        "chunks": repository.list_chunks(document_id),
        "media": repository.list_media_assets(document_id),
        "versions": repository.list_document_versions(document_id),
        "publication_events": repository.list_publication_events(document_id),
    }


@app.post("/api/admin/documents", dependencies=[Depends(require_admin)])
async def upload_documents(
    request: Request,
    response: Response,
    files: Annotated[list[UploadFile], File()],
    knowledge_domain: Annotated[str, Form()] = "company_public",
) -> dict[str, object]:
    if knowledge_domain not in {"company_public", "product_public"}:
        for upload in files:
            await upload.close()
        raise HTTPException(status_code=422, detail="知识域无效")
    if len(files) > settings.max_upload_files:
        for upload in files:
            await upload.close()
        raise _problem_exception(
            413,
            "TOO_MANY_UPLOAD_FILES",
            f"单次最多上传 {settings.max_upload_files} 个文件。",
        )
    maximum_file_bytes = settings.max_file_mb * 1024 * 1024
    maximum_request_bytes = settings.max_upload_request_mb * 1024 * 1024
    known_sizes = [upload.size for upload in files if upload.size is not None]
    if any(size > maximum_file_bytes for size in known_sizes):
        for upload in files:
            await upload.close()
        raise _problem_exception(
            413,
            "UPLOAD_FILE_TOO_LARGE",
            f"单个文件不得超过 {settings.max_file_mb}MB。",
        )
    if sum(known_sizes) > maximum_request_bytes:
        for upload in files:
            await upload.close()
        raise _problem_exception(
            413,
            "UPLOAD_BATCH_TOO_LARGE",
            f"单次上传总量不得超过 {settings.max_upload_request_mb}MB。",
        )

    async def process_uploads() -> dict[str, object]:
        results: list[dict[str, object]] = []
        copied_bytes = 0
        for upload in files:
            filename = safe_filename(upload.filename or "document")
            extension = Path(filename).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                results.append({"filename": filename, "ok": False, "error": "不支持的文件类型"})
                continue
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as handle:
                    temp_path = Path(handle.name)
                copied_bytes += await _copy_upload_limited(upload, temp_path)
                if copied_bytes > maximum_request_bytes:
                    raise _problem_exception(
                        413,
                        "UPLOAD_BATCH_TOO_LARGE",
                        f"单次上传总量不得超过 {settings.max_upload_request_mb}MB。",
                    )
                result = await knowledge_service.ingest_path(
                    temp_path,
                    title=Path(filename).stem,
                    knowledge_domain=knowledge_domain,
                )
                results.append({"filename": filename, "ok": True, **result})
            except HTTPException:
                raise
            except (ServiceOverloadedError, OllamaUnavailableError):
                raise
            except Exception as exc:
                results.append({"filename": filename, "ok": False, "error": str(exc)})
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return {"items": results}

    try:
        result, admission = await _run_admin_task(request, process_uploads)
    finally:
        for upload in files:
            await upload.close()
    _set_admission_header(response, admission)
    return result


@app.patch("/api/admin/documents/{document_id}", dependencies=[Depends(require_admin)])
async def update_document(
    request: Request,
    response: Response,
    document_id: str,
    payload: DocumentUpdate,
) -> dict[str, object]:
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="没有需要更新的字段")
    try:
        document, admission = await _run_admin_task(
            request,
            lambda: asyncio.to_thread(knowledge_service.update_document, document_id, values),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    _set_admission_header(response, admission)
    return {"document": document}


@app.post("/api/admin/documents/{document_id}/reindex", dependencies=[Depends(require_admin)])
async def reindex_document(
    request: Request,
    response: Response,
    document_id: str,
) -> dict[str, object]:
    try:
        document, admission = await _run_admin_task(
            request,
            lambda: knowledge_service.reindex_document(document_id),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _set_admission_header(response, admission)
    return {"document": document}


@app.post("/api/admin/documents/{document_id}/extract-media", dependencies=[Depends(require_admin)])
async def extract_document_media(
    request: Request,
    response: Response,
    document_id: str,
) -> dict[str, object]:
    try:
        result, admission = await _run_admin_task(
            request,
            lambda: knowledge_service.extract_media_with_diff(document_id),
        )
        assets, diff = result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"图片提取失败：{exc}") from exc
    _set_admission_header(response, admission)
    return {"items": assets, "count": len(assets), "diff": diff}


@app.post(
    "/api/admin/documents/{document_id}/rollback/{version}",
    dependencies=[Depends(require_admin)],
)
async def rollback_document(document_id: str, version: int) -> dict[str, object]:
    try:
        repository.rollback_document(document_id, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    document = repository.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document": document,
        "versions": repository.list_document_versions(document_id),
        "publication_events": repository.list_publication_events(document_id),
    }


@app.get("/api/admin/media/{asset_id}/content", dependencies=[Depends(require_admin)])
async def get_admin_media(asset_id: str) -> FileResponse:
    asset = repository.get_media_asset(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="图片不存在")
    return media_file_response(asset)


@app.patch("/api/admin/media/{asset_id}", dependencies=[Depends(require_admin)])
async def update_media(asset_id: str, payload: MediaAssetUpdate) -> dict[str, object]:
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=422, detail="没有需要更新的字段")
    try:
        asset = knowledge_service.update_media_asset(asset_id, values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="图片不存在") from exc
    return {"asset": asset}


@app.patch("/api/admin/chunks/{chunk_id}", dependencies=[Depends(require_admin)])
async def update_chunk(
    request: Request,
    response: Response,
    chunk_id: str,
    payload: ChunkUpdate,
) -> dict[str, object]:
    if payload.content is None and payload.enabled is None:
        raise HTTPException(status_code=422, detail="没有需要更新的字段")
    try:
        chunk, admission = await _run_admin_task(
            request,
            lambda: knowledge_service.update_chunk(chunk_id, payload.content, payload.enabled),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="知识片段不存在") from exc
    except HTTPException:
        raise
    _set_admission_header(response, admission)
    return {"chunk": chunk}


@app.delete("/api/admin/documents/{document_id}", dependencies=[Depends(require_admin)])
async def delete_document(
    request: Request,
    response: Response,
    document_id: str,
) -> dict[str, object]:
    deleted, admission = await _run_admin_task(
        request,
        lambda: knowledge_service.delete_document(document_id),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    _set_admission_header(response, admission)
    return {"deleted": True, "recoverable": True}
