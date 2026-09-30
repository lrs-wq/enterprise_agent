from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

ResultT = TypeVar("ResultT")


class ServiceOverloadedError(RuntimeError):
    def __init__(self, resource: str, retry_after_seconds: int) -> None:
        super().__init__(f"{resource} 当前繁忙")
        self.resource = resource
        self.retry_after_seconds = retry_after_seconds


class ClientDisconnectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapacityAdmission:
    resource: str
    queued: bool
    wait_ms: int


@dataclass(frozen=True)
class CapacitySnapshot:
    resource: str
    active: int
    queued: int
    max_active: int
    max_queue: int


class CapacityGate:
    """Small in-process admission gate with a bounded FIFO wait queue."""

    def __init__(
        self,
        resource: str,
        *,
        max_active: int,
        max_queue: int,
        retry_after_seconds: int = 5,
    ) -> None:
        if max_active < 1 or max_queue < 0:
            raise ValueError("容量参数无效")
        self.resource = resource
        self.max_active = max_active
        self.max_queue = max_queue
        self.retry_after_seconds = retry_after_seconds
        self._tokens: asyncio.Queue[object] = asyncio.Queue(maxsize=max_active)
        for _ in range(max_active):
            self._tokens.put_nowait(object())
        self._active = 0
        self._queued = 0

    def snapshot(self) -> CapacitySnapshot:
        return CapacitySnapshot(
            resource=self.resource,
            active=self._active,
            queued=self._queued,
            max_active=self.max_active,
            max_queue=self.max_queue,
        )

    async def acquire(
        self,
        *,
        deadline: float | None = None,
        on_queued: Callable[[], None] | None = None,
    ) -> CapacityAdmission:
        loop = asyncio.get_running_loop()
        started = loop.time()
        queued = self._tokens.empty()
        if queued:
            if self._queued >= self.max_queue:
                raise ServiceOverloadedError(self.resource, self.retry_after_seconds)
            self._queued += 1
        try:
            if queued and on_queued is not None:
                on_queued()
            if deadline is None:
                await self._tokens.get()
            else:
                if deadline <= loop.time():
                    raise TimeoutError
                async with asyncio.timeout_at(deadline):
                    await self._tokens.get()
        finally:
            if queued:
                self._queued -= 1
        self._active += 1
        return CapacityAdmission(
            resource=self.resource,
            queued=queued,
            wait_ms=max(0, round((loop.time() - started) * 1000)),
        )

    def release(self) -> None:
        if self._active < 1:
            raise RuntimeError(f"{self.resource} 容量槽位重复释放")
        self._active -= 1
        self._tokens.put_nowait(object())

    @asynccontextmanager
    async def slot(
        self,
        *,
        deadline: float | None = None,
        on_queued: Callable[[], None] | None = None,
    ) -> AsyncIterator[CapacityAdmission]:
        admission = await self.acquire(deadline=deadline, on_queued=on_queued)
        try:
            yield admission
        finally:
            self.release()


class SharedCapacityGate(CapacityGate):
    """SQLite-backed FIFO capacity shared by workers on the same host.

    Active leases are renewed while a slot is held, so a killed worker cannot
    permanently consume capacity. The database must be on a local filesystem.
    """

    def __init__(
        self,
        resource: str,
        *,
        database_path: Path,
        max_active: int,
        max_queue: int,
        retry_after_seconds: int = 5,
        lease_seconds: float = 45,
        poll_seconds: float = 0.05,
    ) -> None:
        super().__init__(
            resource,
            max_active=max_active,
            max_queue=max_queue,
            retry_after_seconds=retry_after_seconds,
        )
        self.database_path = database_path
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self._owned: dict[asyncio.Task[Any], list[str]] = {}
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capacity_leases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket TEXT NOT NULL UNIQUE,
                    resource TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('active', 'queued')),
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_capacity_resource_state "
                "ON capacity_leases(resource, state, id)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=1, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=1000")
        return connection

    def _transact(self, operation: Callable[[sqlite3.Connection], ResultT]) -> ResultT:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM capacity_leases WHERE expires_at <= ?", (time.time(),)
                )
                result = operation(connection)
                connection.execute("COMMIT")
                return result
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def snapshot(self) -> CapacitySnapshot:
        def query(connection: sqlite3.Connection) -> tuple[int, int]:
            rows = connection.execute(
                "SELECT state, COUNT(*) FROM capacity_leases WHERE resource = ? GROUP BY state",
                (self.resource,),
            ).fetchall()
            counts = dict(rows)
            return counts.get("active", 0), counts.get("queued", 0)

        active, queued = self._transact(query)
        return CapacitySnapshot(self.resource, active, queued, self.max_active, self.max_queue)

    def _admit(self, ticket: str) -> str:
        def insert(connection: sqlite3.Connection) -> str:
            active = connection.execute(
                "SELECT COUNT(*) FROM capacity_leases WHERE resource = ? AND state = 'active'",
                (self.resource,),
            ).fetchone()[0]
            queued = connection.execute(
                "SELECT COUNT(*) FROM capacity_leases WHERE resource = ? AND state = 'queued'",
                (self.resource,),
            ).fetchone()[0]
            if active < self.max_active and queued == 0:
                state = "active"
            elif queued < self.max_queue:
                state = "queued"
            else:
                raise ServiceOverloadedError(self.resource, self.retry_after_seconds)
            connection.execute(
                "INSERT INTO capacity_leases(ticket, resource, state, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (ticket, self.resource, state, time.time() + self.lease_seconds),
            )
            return state

        return self._transact(insert)

    def _promote(self, ticket: str) -> bool:
        def update(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT id, state FROM capacity_leases WHERE ticket = ? AND resource = ?",
                (ticket, self.resource),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"{self.resource} 排队租约已失效")
            connection.execute(
                "UPDATE capacity_leases SET expires_at = ? WHERE ticket = ?",
                (time.time() + self.lease_seconds, ticket),
            )
            if row[1] == "active":
                return True
            head = connection.execute(
                "SELECT id FROM capacity_leases WHERE resource = ? AND state = 'queued' "
                "ORDER BY id LIMIT 1",
                (self.resource,),
            ).fetchone()
            active = connection.execute(
                "SELECT COUNT(*) FROM capacity_leases WHERE resource = ? AND state = 'active'",
                (self.resource,),
            ).fetchone()[0]
            if head is None or head[0] != row[0] or active >= self.max_active:
                return False
            connection.execute(
                "UPDATE capacity_leases SET state = 'active' WHERE ticket = ?", (ticket,)
            )
            return True

        return self._transact(update)

    def _remove(self, ticket: str) -> None:
        self._transact(
            lambda connection: connection.execute(
                "DELETE FROM capacity_leases WHERE ticket = ? AND resource = ?",
                (ticket, self.resource),
            )
        )

    async def acquire(
        self,
        *,
        deadline: float | None = None,
        on_queued: Callable[[], None] | None = None,
    ) -> CapacityAdmission:
        loop = asyncio.get_running_loop()
        started = loop.time()
        if deadline is not None and deadline <= started:
            raise TimeoutError
        ticket = uuid.uuid4().hex
        state = self._admit(ticket)
        try:
            if state == "queued":
                if on_queued is not None:
                    on_queued()
                while True:
                    if deadline is not None and loop.time() >= deadline:
                        raise TimeoutError
                    if self._promote(ticket):
                        break
                    delay = self.poll_seconds
                    if deadline is not None:
                        delay = min(delay, max(0, deadline - loop.time()))
                    await asyncio.sleep(delay)
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("容量槽位只能由异步任务持有")
            self._owned.setdefault(task, []).append(ticket)
            return CapacityAdmission(
                self.resource,
                state == "queued",
                max(0, round((loop.time() - started) * 1000)),
            )
        except BaseException:
            self._remove(ticket)
            raise

    def release(self) -> None:
        task = asyncio.current_task()
        tickets = self._owned.get(task) if task is not None else None
        if not tickets:
            raise RuntimeError(f"{self.resource} 容量槽位重复释放")
        ticket = tickets[-1]
        self._remove(ticket)
        tickets.pop()
        if not tickets:
            assert task is not None
            del self._owned[task]

    async def _heartbeat(self, ticket: str, owner: asyncio.Task[Any]) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 4)

            def renew(connection: sqlite3.Connection) -> bool:
                cursor = connection.execute(
                    "UPDATE capacity_leases SET expires_at = ? "
                    "WHERE ticket = ? AND resource = ? AND state = 'active'",
                    (time.time() + self.lease_seconds, ticket, self.resource),
                )
                return cursor.rowcount == 1

            try:
                renewed = self._transact(renew)
            except sqlite3.Error:
                continue
            if not renewed:
                owner.cancel()
                return

    @asynccontextmanager
    async def slot(
        self,
        *,
        deadline: float | None = None,
        on_queued: Callable[[], None] | None = None,
    ) -> AsyncIterator[CapacityAdmission]:
        admission = await self.acquire(deadline=deadline, on_queued=on_queued)
        task = asyncio.current_task()
        assert task is not None
        heartbeat = asyncio.create_task(self._heartbeat(self._owned[task][-1], task))
        try:
            yield admission
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            self.release()


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


class SlidingWindowLimiter:
    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1 or window_seconds < 1 or max_keys < 1:
            raise ValueError("限速参数无效")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._last_sweep = clock()

    @property
    def tracked_keys(self) -> int:
        return len(self._events)

    def _sweep(self, now: float, *, force: bool = False) -> None:
        sweep_interval = min(float(self.window_seconds), 60.0)
        if not force and now - self._last_sweep < sweep_interval:
            return
        cutoff = now - self.window_seconds
        for key, events in tuple(self._events.items()):
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)
        self._last_sweep = now

    def _make_room(self, now: float) -> None:
        if len(self._events) < self.max_keys:
            return
        self._sweep(now, force=True)
        if len(self._events) < self.max_keys:
            return
        oldest_key = min(
            self._events,
            key=lambda key: self._events[key][-1] if self._events[key] else float("-inf"),
        )
        self._events.pop(oldest_key, None)

    def _current(self, key: str) -> tuple[deque[float], float]:
        now = self._clock()
        self._sweep(now)
        events = self._events.get(key)
        if events is None:
            self._make_room(now)
            events = deque()
            self._events[key] = events
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if not events:
            self._events.pop(key, None)
            events = self._events.setdefault(key, deque())
        return events, now

    def retry_after(self, key: str) -> int | None:
        events, now = self._current(key)
        if len(events) < self.limit:
            return None
        return max(1, int(events[0] + self.window_seconds - now + 0.999))

    def consume(self, key: str) -> RateLimitDecision:
        events, now = self._current(key)
        if len(events) >= self.limit:
            retry_after = max(1, int(events[0] + self.window_seconds - now + 0.999))
            return RateLimitDecision(False, retry_after)
        events.append(now)
        return RateLimitDecision(True, 0)

    def record(self, key: str) -> None:
        events, now = self._current(key)
        events.append(now)

    def reset(self, key: str) -> None:
        self._events.pop(key, None)


async def run_until_disconnected(
    operation: Awaitable[ResultT],
    wait_for_disconnect: Callable[[], Coroutine[Any, Any, None]],
) -> ResultT:
    """Cancel an operation when its HTTP client goes away."""

    async def await_operation() -> ResultT:
        return await operation

    operation_task = asyncio.create_task(await_operation())
    disconnect_task = asyncio.create_task(wait_for_disconnect())
    try:
        done, _ = await asyncio.wait(
            {operation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task
        operation_task.cancel()
        with suppress(asyncio.CancelledError):
            await operation_task
        raise ClientDisconnectedError("客户端已断开连接")
    finally:
        if not operation_task.done():
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
        if not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
