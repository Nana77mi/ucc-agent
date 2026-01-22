from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


Message = Dict[str, str]


@dataclass
class SessionState:
    """单个会话的内存状态。"""
    messages: List[Message]
    expire_at: float


class MemoryStore:
    """会话存储抽象接口。"""
    def get(self, session_id: str) -> List[Message]:
        raise NotImplementedError

    def append(self, session_id: str, new_messages: List[Message]) -> List[Message]:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError

    def touch(self, session_id: str) -> None:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """基于内存的会话存储。"""
    def __init__(self, *, ttl_seconds: int = 1800, max_messages: int = 20) -> None:
        # 保存 TTL 与最大消息数
        self._ttl_seconds = ttl_seconds
        self._max_messages = max_messages
        self._store: Dict[str, SessionState] = {}

    def _now(self) -> float:
        """返回当前时间戳。"""
        return time.time()

    def _expire_at(self) -> float:
        """计算过期时间。"""
        return self._now() + self._ttl_seconds

    def _prune_expired(self) -> None:
        """清理过期会话。"""
        now = self._now()
        expired = [sid for sid, state in self._store.items() if state.expire_at <= now]
        for sid in expired:
            self._store.pop(sid, None)

    def _trim_messages(self, messages: List[Message]) -> List[Message]:
        """截断消息列表到最大长度。"""
        if self._max_messages <= 0:
            return messages
        return messages[-self._max_messages :]

    def get(self, session_id: str) -> List[Message]:
        """获取会话历史。"""
        self._prune_expired()
        state = self._store.get(session_id)
        if not state:
            return []
        if state.expire_at <= self._now():
            self._store.pop(session_id, None)
            return []
        return list(state.messages)

    def append(self, session_id: str, new_messages: List[Message]) -> List[Message]:
        """追加消息并返回最新历史。"""
        self._prune_expired()
        existing = self._store.get(session_id)
        messages = list(existing.messages) if existing else []
        messages.extend(new_messages)
        messages = self._trim_messages(messages)
        self._store[session_id] = SessionState(messages=messages, expire_at=self._expire_at())
        return list(messages)

    def clear(self, session_id: str) -> None:
        """清空会话内容。"""
        self._store.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        """刷新会话过期时间。"""
        self._prune_expired()
        state = self._store.get(session_id)
        if not state:
            return
        self._store[session_id] = SessionState(messages=state.messages, expire_at=self._expire_at())

    def stats(self) -> Dict[str, Optional[int]]:
        """返回统计信息。"""
        self._prune_expired()
        return {
            "sessions": len(self._store),
            "ttl_seconds": self._ttl_seconds,
            "max_messages": self._max_messages,
        }
