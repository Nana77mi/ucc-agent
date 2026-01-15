from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from src.common import load_yaml, read_text
from src.model_factory import build_embeddings, build_llm
from src.rag_retrieval import CrossEncoderReranker, retrieve_ranked_docs


Message = Dict[str, str]


@dataclass(frozen=True)
class ChatMetrics:
    elapsed: float
    total_tokens: Optional[int]
    tokens_per_s: Optional[float]


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    docs: List[Document]
    metrics: Optional[ChatMetrics]


@dataclass(frozen=True)
class ChatSettings:
    persist_dir: str
    embed_model: str
    llm_model: str
    top_k: int
    show_k: int
    max_ctx: int
    score_threshold: float
    keyword_boost: float
    system_prompt_path: str
    temperature: float


def safe_meta(doc: Document) -> Dict[str, Any]:
    return doc.metadata or {}


def build_context_block(docs: List[Document]) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        m = safe_meta(d)
        src = m.get("source", "")
        doc_id = m.get("id", "")
        title = m.get("title", "")
        section = m.get("section", "")
        chunk_id = m.get("chunk_id", None)
        chunk_total = m.get("chunk_total", None)

        head = f"[{i}] source={src} id={doc_id}"
        if title:
            head += f" title={title}"
        if section:
            head += f" section={section}"
        if chunk_id is not None and chunk_total is not None:
            head += f" chunk={chunk_id}/{chunk_total}"

        parts.append(head + "\n" + (d.page_content or "").strip())
    return "\n\n---\n\n".join(parts)


def build_user_prompt(user_query: str, context_block: str) -> str:
    return f"""你是文档问答助手。

请严格基于下方 Context 回答问题，并在回答中用 [1][2] 的形式标注引用来源。
如果无法从 Context 得到答案，请回答“文档中未找到相关信息”。

【Context】
{context_block}

【Question】
{user_query}

【Answer】
"""


def build_messages(
    system_prompt: str,
    user_query: str,
    context_block: str,
    history: Optional[Iterable[Message]] = None,
) -> List[Message]:
    sys = (system_prompt or "").strip()
    user = build_user_prompt(user_query, context_block)
    messages: List[Message] = [{"role": "system", "content": sys}]
    if history:
        for item in history:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def extract_total_tokens(resp: Any) -> Optional[int]:
    usage = getattr(resp, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
            return int(prompt_tokens or 0) + int(completion_tokens or 0)

    meta = getattr(resp, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(token_usage, dict):
            total = token_usage.get("total_tokens")
            if isinstance(total, int):
                return total
            prompt_tokens = token_usage.get("prompt_tokens")
            completion_tokens = token_usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
                return int(prompt_tokens or 0) + int(completion_tokens or 0)

        prompt_eval = meta.get("prompt_eval_count")
        eval_count = meta.get("eval_count")
        if isinstance(prompt_eval, int) or isinstance(eval_count, int):
            return int(prompt_eval or 0) + int(eval_count or 0)

    return None


class ChatEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self._cfg = cfg
        self.settings = self._load_settings(cfg)
        self.system_prompt = read_text(
            self.settings.system_prompt_path,
            default="你是 UCC 文档助手。你只能基于已检索到的片段回答，禁止猜测。",
        )
        self._reranker = self._build_reranker(cfg)
        if not self._reranker:
            self._reranker = None

        if not self.settings.persist_dir:
            raise FileNotFoundError("FAISS index dir not found: empty path")
        self.embeddings = build_embeddings(cfg)
        self.db = FAISS.load_local(
            self.settings.persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = build_llm(cfg, temperature=self.settings.temperature)

    @classmethod
    def from_yaml(cls, path: str) -> "ChatEngine":
        cfg = load_yaml(path)
        return cls(cfg)

    @staticmethod
    def _load_settings(cfg: Dict[str, Any]) -> ChatSettings:
        persist_dir = cfg.get("paths", {}).get("persist_dir", "index/faiss_ucc")
        embed_model = cfg.get("models", {}).get("embed_model", "nomic-embed-text:latest")
        llm_model = cfg.get("models", {}).get("llm_model", "qwen3:4b")

        retrieval_cfg = cfg.get("retrieval", {}) or {}
        top_k = int(retrieval_cfg.get("top_k", 20))
        show_k = int(retrieval_cfg.get("show_k", top_k))

        runtime_cfg = cfg.get("runtime", {}) or {}
        max_ctx = int(runtime_cfg.get("max_context_docs", 10))

        rag_cfg = cfg.get("rag", {}) or {}
        score_threshold = float(rag_cfg.get("score_threshold", 0.0))
        keyword_boost = float(rag_cfg.get("keyword_boost", 0.25))

        temperature = float(cfg.get("generation", {}).get("temperature", runtime_cfg.get("temperature", 0.2)))

        system_prompt_path = cfg.get("paths", {}).get("system_prompt", "prompts/system.txt")

        return ChatSettings(
            persist_dir=persist_dir,
            embed_model=embed_model,
            llm_model=llm_model,
            top_k=top_k,
            show_k=show_k,
            max_ctx=max_ctx,
            score_threshold=score_threshold,
            keyword_boost=keyword_boost,
            system_prompt_path=system_prompt_path,
            temperature=temperature,
        )

    @staticmethod
    def _build_reranker(cfg: Dict[str, Any]) -> Optional[CrossEncoderReranker]:
        rerank_cfg = cfg.get("rerank", {}) or {}
        if not bool(rerank_cfg.get("enabled", False)):
            return None
        rerank_model = str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))
        rerank_batch_size = int(rerank_cfg.get("batch_size", 16))
        rerank_cache_dir = str(rerank_cfg.get("cache_dir", "models/rerank"))
        reranker = CrossEncoderReranker(
            rerank_model,
            batch_size=rerank_batch_size,
            cache_dir=rerank_cache_dir,
        )
        return reranker if reranker.available else None

    def retrieve_docs(self, query: str) -> List[Document]:
        rerank_cfg = self._cfg.get("rerank", {}) or {}
        rerank_top_n = int(rerank_cfg.get("top_n", self.settings.top_k))
        return retrieve_ranked_docs(
            db=self.db,
            query=query,
            top_k=self.settings.top_k,
            score_threshold=self.settings.score_threshold,
            keyword_boost=self.settings.keyword_boost,
            reranker=self._reranker,
            rerank_top_n=rerank_top_n,
            llm=self.llm,
            embeddings=self.embeddings,
            cfg=self._cfg,
        )

    def answer(self, query: str, history: Optional[Iterable[Message]] = None) -> ChatResponse:
        docs_ranked = self.retrieve_docs(query)
        if not docs_ranked:
            return ChatResponse(
                answer="文档中未找到相关信息",
                docs=[],
                metrics=None,
            )

        docs_for_llm = docs_ranked[: self.settings.max_ctx]
        context_block = build_context_block(docs_for_llm)
        messages = build_messages(self.system_prompt, query, context_block, history)

        start_time = time.perf_counter()
        resp = self.llm.invoke(messages)
        elapsed = time.perf_counter() - start_time
        ans = getattr(resp, "content", str(resp))
        total_tokens = extract_total_tokens(resp)
        tokens_per_s = total_tokens / elapsed if total_tokens and elapsed > 0 else None
        metrics = ChatMetrics(elapsed=elapsed, total_tokens=total_tokens, tokens_per_s=tokens_per_s)
        return ChatResponse(answer=(ans or "").strip(), docs=docs_ranked, metrics=metrics)

    def reranker_status(self) -> Optional[str]:
        if not self._reranker:
            return None
        return "enabled"
