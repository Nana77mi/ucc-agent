from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


Message = Dict[str, str]


@dataclass
class SessionState:
    messages: List[Message]
    expire_at: float


class MemoryStore:
    def get(self, session_id: str) -> List[Message]:
        raise NotImplementedError

    def append(self, session_id: str, new_messages: List[Message]) -> List[Message]:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError

    def touch(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    def __init__(self, *, ttl_seconds: int = 1800, max_messages: int = 20) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_messages = max_messages
        self._store: Dict[str, SessionState] = {}

    def _now(self) -> float:
        return time.time()

    def _expire_at(self) -> float:
        return self._now() + self._ttl_seconds

    def _prune_expired(self) -> None:
        now = self._now()
        expired = [sid for sid, state in self._store.items() if state.expire_at <= now]
        for sid in expired:
            self._store.pop(sid, None)

    def _trim_messages(self, messages: List[Message]) -> List[Message]:
        if self._max_messages <= 0:
            return messages
        return messages[-self._max_messages :]

    def get(self, session_id: str) -> List[Message]:
        self._prune_expired()
        state = self._store.get(session_id)
        if not state:
            return []
        if state.expire_at <= self._now():
            self._store.pop(session_id, None)
            return []
        return list(state.messages)

    def append(self, session_id: str, new_messages: List[Message]) -> List[Message]:
        self._prune_expired()
        existing = self._store.get(session_id)
        messages = list(existing.messages) if existing else []
        messages.extend(new_messages)
        messages = self._trim_messages(messages)
        self._store[session_id] = SessionState(messages=messages, expire_at=self._expire_at())
        return list(messages)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        self._prune_expired()
        state = self._store.get(session_id)
        if not state:
            return
        self._store[session_id] = SessionState(messages=state.messages, expire_at=self._expire_at())

    def stats(self) -> Dict[str, Optional[int]]:
        self._prune_expired()
        return {
            "sessions": len(self._store),
            "ttl_seconds": self._ttl_seconds,
            "max_messages": self._max_messages,
        }
