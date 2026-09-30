import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException, Response
from httpx import ASGITransport, AsyncClient

from company_agent import main
from company_agent.main import RequestBodyLimitMiddleware
from company_agent.runtime_control import (
    CapacityGate,
    ClientDisconnectedError,
    ServiceOverloadedError,
    SharedCapacityGate,
    SlidingWindowLimiter,
    run_until_disconnected,
)
from company_agent.services import ModelStageBudgetExceeded


async def wait_for_queued(gate: CapacityGate, expected: int) -> None:
    for _ in range(100):
        if gate.snapshot().queued == expected:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"queue did not reach {expected}")


async def test_capacity_gate_has_bounded_fifo_queue_and_fast_rejection() -> None:
    gate = CapacityGate("test", max_active=1, max_queue=2, retry_after_seconds=3)
    first = await gate.acquire()
    second_task = asyncio.create_task(gate.acquire())
    third_task = asyncio.create_task(gate.acquire())
    await wait_for_queued(gate, 2)

    with pytest.raises(ServiceOverloadedError) as raised:
        await gate.acquire()
    assert raised.value.retry_after_seconds == 3
    assert gate.snapshot().active == 1
    assert gate.snapshot().queued == 2

    gate.release()
    second = await second_task
    assert second.queued is True
    gate.release()
    third = await third_task
    assert third.queued is True
    gate.release()
    assert gate.snapshot().active == 0
    assert gate.snapshot().queued == 0
    assert first.queued is False


async def test_capacity_wait_timeout_and_cancellation_remove_queue_entries() -> None:
    gate = CapacityGate("test", max_active=1, max_queue=1)
    await gate.acquire()
    deadline = asyncio.get_running_loop().time() + 0.01
    with pytest.raises(TimeoutError):
        await gate.acquire(deadline=deadline)
    assert gate.snapshot().queued == 0

    waiter = asyncio.create_task(gate.acquire())
    await wait_for_queued(gate, 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert gate.snapshot().queued == 0
    gate.release()


async def test_queued_progress_callback_error_does_not_leak_capacity() -> None:
    gate = CapacityGate("chat", max_active=1, max_queue=1)
    await gate.acquire()

    def fail_progress() -> None:
        raise ValueError("progress sink failed")

    with pytest.raises(ValueError, match="progress sink failed"):
        await gate.acquire(on_queued=fail_progress)
    assert gate.snapshot().queued == 0
    gate.release()


async def test_cancelled_active_capacity_slot_is_released() -> None:
    gate = CapacityGate("model", max_active=1, max_queue=0)
    entered = asyncio.Event()

    async def long_model_call() -> None:
        async with gate.slot():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(long_model_call())
    await entered.wait()
    assert gate.snapshot().active == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.snapshot().active == 0
    assert gate.snapshot().queued == 0


async def test_shared_capacity_gate_coordinates_workers_and_cancellation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "capacity.db"
    gates = [
        SharedCapacityGate("model", database_path=database_path, max_active=1, max_queue=2)
        for _ in range(4)
    ]
    third_ready = asyncio.Event()
    third_release = asyncio.Event()

    async def third_work() -> None:
        async with gates[2].slot() as admission:
            assert admission.queued is True
            third_ready.set()
            await third_release.wait()

    async with gates[0].slot():
        second = asyncio.create_task(gates[1].acquire())
        third = asyncio.create_task(third_work())
        for _ in range(100):
            if gates[3].snapshot().queued == 2:
                break
            await asyncio.sleep(0.01)
        assert gates[3].snapshot().active == 1
        assert gates[3].snapshot().queued == 2
        with pytest.raises(ServiceOverloadedError):
            await gates[3].acquire()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert gates[3].snapshot().queued == 1

    await asyncio.wait_for(third_ready.wait(), timeout=1)
    assert gates[3].snapshot().active == 1
    third_release.set()
    await third
    assert gates[3].snapshot().active == 0


async def test_shared_capacity_gate_reclaims_expired_worker_lease(tmp_path: Path) -> None:
    database_path = tmp_path / "capacity.db"
    first = SharedCapacityGate(
        "chat",
        database_path=database_path,
        max_active=1,
        max_queue=0,
        lease_seconds=0.05,
    )
    second = SharedCapacityGate(
        "chat",
        database_path=database_path,
        max_active=1,
        max_queue=0,
        lease_seconds=0.05,
    )
    await first.acquire()
    await asyncio.sleep(0.06)
    assert second.snapshot().active == 0
    await second.acquire()
    assert first.snapshot().active == 1
    second.release()
    first.release()


async def test_shared_capacity_gate_enforces_limit_across_processes(tmp_path: Path) -> None:
    database_path = tmp_path / "capacity.db"
    gate = SharedCapacityGate("chat", database_path=database_path, max_active=1, max_queue=0)
    code = (
        "import asyncio, pathlib, sys\n"
        "from company_agent.runtime_control import SharedCapacityGate, ServiceOverloadedError\n"
        "async def run():\n"
        "    gate = SharedCapacityGate('chat', database_path=pathlib.Path(sys.argv[1]), "
        "max_active=1, max_queue=0)\n"
        "    try:\n"
        "        await gate.acquire()\n"
        "    except ServiceOverloadedError:\n"
        "        print('busy')\n"
        "    else:\n"
        "        print('admitted')\n"
        "        gate.release()\n"
        "asyncio.run(run())\n"
    )
    async with gate.slot():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            str(database_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
        assert process.returncode == 0, stderr.decode()
        assert stdout.decode().strip() == "busy"


def test_sliding_window_rate_limiter_returns_retry_after() -> None:
    now = [100.0]
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10, clock=lambda: now[0])

    assert limiter.consume("client").allowed is True
    assert limiter.consume("client").allowed is True
    blocked = limiter.consume("client")
    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 10

    now[0] = 111.0
    assert limiter.consume("client").allowed is True


def test_sliding_window_rate_limiter_bounds_and_expires_client_keys() -> None:
    now = [100.0]
    limiter = SlidingWindowLimiter(
        limit=2,
        window_seconds=10,
        max_keys=2,
        clock=lambda: now[0],
    )

    limiter.consume("first")
    limiter.consume("second")
    limiter.consume("third")
    assert limiter.tracked_keys == 2

    now[0] = 111.0
    limiter.consume("fourth")
    assert limiter.tracked_keys == 1


async def test_disconnect_monitor_cancels_in_flight_operation() -> None:
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()

    async def operation() -> None:
        operation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    async def disconnected() -> None:
        await asyncio.sleep(0.001)

    with pytest.raises(ClientDisconnectedError):
        await run_until_disconnected(
            operation(),
            disconnected,
        )
    assert operation_started.is_set()
    assert operation_cancelled.is_set()


async def test_body_limit_middleware_stops_chunked_request() -> None:
    async def app(
        _scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = iter(
        [
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"cd", "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(app, {"/limited": 3})  # type: ignore[arg-type]
    await middleware(
        {"type": "http", "path": "/limited"},  # type: ignore[arg-type]
        receive,  # type: ignore[arg-type]
        send,  # type: ignore[arg-type]
    )

    assert sent[0]["status"] == 413
    payload = json.loads(sent[1]["body"])
    assert payload["detail"]["code"] == "REQUEST_BODY_TOO_LARGE"


async def test_chat_stream_route_keeps_request_body_limit() -> None:
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat/stream",
            content=b"x" * (main.settings.max_chat_request_kb * 1024 + 1),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "REQUEST_BODY_TOO_LARGE"


async def test_chat_stream_http_route_emits_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_chat(
        _payload: object, *, deadline: float, progress: Callable[[str], None]
    ) -> main.ChatResponse:
        del deadline
        progress("validating")
        decision = main.agent_service.boundary.classify("你好")
        return main.agent_service._fixed_response("http-stream", decision)  # noqa: SLF001

    monkeypatch.setattr(main, "chat_gate", CapacityGate("test-chat", max_active=1, max_queue=0))
    monkeypatch.setattr(main.agent_service, "chat", fake_chat)
    monkeypatch.setattr(main.repository, "update_query_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main,
        "chat_rate_limiter",
        SlidingWindowLimiter(limit=100, window_seconds=60),
    )
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat/stream", json={"question": "你好", "history": []})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.index("event: progress\n") < response.text.index("event: result\n")
    assert '"request_id": "http-stream"' in response.text


async def test_chat_total_deadline_returns_retryable_504(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        host = "198.51.100.50"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

        async def receive(self) -> dict[str, str]:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

    async def slow_chat(_payload: object, *, deadline: float) -> None:
        del deadline
        await asyncio.sleep(10)

    monkeypatch.setattr(main.settings, "chat_request_timeout_seconds", 0.01)
    monkeypatch.setattr(main.agent_service, "chat", slow_chat)

    with pytest.raises(HTTPException) as raised:
        await main.chat(  # type: ignore[arg-type]
            FakeRequest(),
            Response(),
            main.ChatRequest(question="测试超时"),
        )

    assert raised.value.status_code == 504
    assert raised.value.detail["code"] == "CHAT_DEADLINE_EXCEEDED"


async def test_chat_stream_reports_real_stages_before_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        host = "198.51.100.80"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

    async def fake_chat(
        _payload: object,
        *,
        deadline: float,
        progress: Callable[[str], None],
    ) -> main.ChatResponse:
        del deadline
        for stage in ("retrieving", "generating", "validating"):
            progress(stage)
            await asyncio.sleep(0)
        decision = main.agent_service.boundary.classify("你好")
        return main.agent_service._fixed_response("stream-test", decision)  # noqa: SLF001

    monkeypatch.setattr(main, "chat_gate", CapacityGate("test-chat", max_active=1, max_queue=1))
    monkeypatch.setattr(main.agent_service, "chat", fake_chat)
    monkeypatch.setattr(main.repository, "update_query_runtime", lambda *_args, **_kwargs: None)

    response = await main.chat_stream(  # type: ignore[arg-type]
        FakeRequest(), main.ChatRequest(question="你好")
    )
    body = "".join([part async for part in response.body_iterator])

    assert body.startswith("event: progress\n")
    assert '"stage": "accepted"' in body
    assert '"stage": "retrieving"' in body
    assert '"stage": "validating"' in body
    assert body.index('"stage": "validating"') < body.index("event: result\n")
    assert "你好" not in body.split("event: result\n", 1)[0]
    assert response.headers["x-accel-buffering"] == "no"


async def test_chat_stream_finishes_without_waiting_for_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        host = "198.51.100.82"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

    async def immediate_chat(
        _payload: object, *, deadline: float, progress: Callable[[str], None]
    ) -> main.ChatResponse:
        del deadline, progress
        decision = main.agent_service.boundary.classify("你好")
        return main.agent_service._fixed_response("stream-fast", decision)  # noqa: SLF001

    monkeypatch.setattr(main, "chat_gate", CapacityGate("test-chat", max_active=1, max_queue=0))
    monkeypatch.setattr(main.agent_service, "chat", immediate_chat)
    monkeypatch.setattr(main.repository, "update_query_runtime", lambda *_args, **_kwargs: None)
    response = await main.chat_stream(  # type: ignore[arg-type]
        FakeRequest(), main.ChatRequest(question="你好")
    )

    async def collect() -> str:
        return "".join([part async for part in response.body_iterator])

    body = await asyncio.wait_for(collect(), timeout=0.5)
    assert "event: result\n" in body


async def test_chat_stream_busy_error_uses_event_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        host = "198.51.100.83"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

    gate = CapacityGate("test-chat", max_active=1, max_queue=0)
    await gate.acquire()
    monkeypatch.setattr(main, "chat_gate", gate)
    response = await main.chat_stream(  # type: ignore[arg-type]
        FakeRequest(), main.ChatRequest(question="测试繁忙")
    )
    body = "".join([part async for part in response.body_iterator])
    gate.release()

    assert "event: error\n" in body
    assert '"code": "CHAT_BUSY"' in body
    assert '"status": 429' in body


async def test_chat_stream_emits_queue_and_cancels_work_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        host = "198.51.100.81"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

    cancelled = asyncio.Event()

    async def fake_chat(
        _payload: object,
        *,
        deadline: float,
        progress: Callable[[str], None],
    ) -> None:
        del deadline
        progress("generating")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    gate = CapacityGate("test-chat", max_active=1, max_queue=1)
    await gate.acquire()
    monkeypatch.setattr(main, "chat_gate", gate)
    monkeypatch.setattr(main.agent_service, "chat", fake_chat)
    response = await main.chat_stream(  # type: ignore[arg-type]
        FakeRequest(), main.ChatRequest(question="测试排队")
    )
    iterator = response.body_iterator

    assert '"stage": "accepted"' in await asyncio.wait_for(anext(iterator), timeout=0.5)
    assert '"stage": "queued"' in await asyncio.wait_for(anext(iterator), timeout=1)
    gate.release()
    assert '"stage": "generating"' in await asyncio.wait_for(anext(iterator), timeout=1)
    await iterator.aclose()

    assert cancelled.is_set()
    assert gate.snapshot().active == 0


async def test_chat_model_stage_budget_returns_specific_504(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        host = "198.51.100.51"

    class FakeRequest:
        client = FakeClient()
        headers: dict[str, str] = {}

        async def receive(self) -> dict[str, str]:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

    async def timed_out_chat(_payload: object, *, deadline: float) -> None:
        del deadline
        raise ModelStageBudgetExceeded("证据修复")

    monkeypatch.setattr(main.agent_service, "chat", timed_out_chat)

    with pytest.raises(HTTPException) as raised:
        await main.chat(  # type: ignore[arg-type]
            FakeRequest(),
            Response(),
            main.ChatRequest(question="测试模型阶段超时"),
        )

    assert raised.value.status_code == 504
    assert raised.value.detail["code"] == "MODEL_STAGE_TIMEOUT"
