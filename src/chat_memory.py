from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.memory_store import InMemoryStore, MemoryStore, Message


@dataclass(frozen=True)
class MemoryConfig:
    ttl_seconds: int = 1800
    max_messages: int = 20


class MemoryManager:
    def __init__(self, store: MemoryStore, config: MemoryConfig) -> None:
        self._store = store
        self._config = config

    @classmethod
    def in_memory(cls, config: MemoryConfig | None = None) -> "MemoryManager":
        cfg = config or MemoryConfig()
        store = InMemoryStore(ttl_seconds=cfg.ttl_seconds, max_messages=cfg.max_messages)
        return cls(store=store, config=cfg)

    def get_history(self, session_id: str) -> List[Message]:
        return self._store.get(session_id)

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> List[Message]:
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        return self._store.append(session_id, messages)

    def clear(self, session_id: str) -> None:
        self._store.clear(session_id)

    def touch(self, session_id: str) -> None:
        self._store.touch(session_id)

    def stats(self) -> Dict[str, int | None]:
        stats = {}
        if hasattr(self._store, "stats"):
            stats = self._store.stats()  # type: ignore[assignment]
        return {
            "sessions": stats.get("sessions"),
            "ttl_seconds": self._config.ttl_seconds,
            "max_messages": self._config.max_messages,
        }

    @staticmethod
    def merge_history(
        base: Iterable[Message] | None,
        extra: Iterable[Message] | None,
    ) -> List[Message]:
        merged: List[Message] = []
        for group in (base or [], extra or []):
            for item in group:
                role = str(item.get("role", "")).strip()
                content = str(item.get("content", "")).strip()
                if role and content:
                    merged.append({"role": role, "content": content})
        return merged
