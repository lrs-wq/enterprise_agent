"""Small in-process cache for already validated, public, context-free answers."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from company_agent.schemas import ChatResponse


@dataclass(frozen=True)
class _Entry:
    response: ChatResponse
    expires_at: float


class PublicAnswerCache:
    def __init__(self, *, max_entries: int, ttl_seconds: int) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def get(self, key: str) -> ChatResponse | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry.response.model_copy(deep=True)

    def put(self, key: str, response: ChatResponse) -> None:
        self._entries[key] = _Entry(
            response=response.model_copy(deep=True),
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def discard(self, key: str) -> None:
        self._entries.pop(key, None)
