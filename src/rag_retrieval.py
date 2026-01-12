# src/rag_retrieval.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import os
import time
import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import FAISS

from src.query_rewrite import QueryCandidate, rewrite_queries


def safe_meta(doc: Document) -> Dict[str, Any]:
    return doc.metadata or {}


def apply_score_threshold(pairs: List[Tuple[Document, float]], score_threshold: float) -> List[Tuple[Document, float]]:
    # 与 02_chat.py 保持一致：L2 距离越小越相似
    try:
        thr = float(score_threshold)
    except Exception:
        return pairs
    if thr <= 0:
        return pairs
    return [(d, s) for (d, s) in pairs if s <= thr]


def doc_contains_query(query: str, doc: Document) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    md = doc.metadata or {}
    hay = " ".join(
        [
            str(md.get("title") or ""),
            str(md.get("section") or ""),
            str(md.get("source") or ""),
            " ".join([str(t) for t in (md.get("tags") or [])]),
            (doc.page_content or "")[:1500],
        ]
    ).lower()
    return q in hay


def rerank_with_keyword_boost(query: str, results: List[Tuple[Document, float]], boost: float = 0.25) -> List[Tuple[Document, float]]:
    out: List[Tuple[Document, float]] = []
    for d, s in results:
        s2 = (s - boost) if doc_contains_query(query, d) else s
        out.append((d, s2))
    out.sort(key=lambda x: x[1])  # 距离越小越前
    return out


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.available = False
        self._init_error: Optional[Exception] = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore
            resolved_model = self._resolve_model_path(model_name, cache_dir, snapshot_download)
            self._ce = CrossEncoder(resolved_model)
            self.available = True
        except Exception as e:
            self._ce = None
            self.available = False
            self._init_error = e

    @staticmethod
    def _resolve_model_path(
        model_name: str,
        cache_dir: Optional[str],
        snapshot_download,
    ) -> str:
        if os.path.isdir(model_name):
            return model_name
        if not cache_dir:
            return model_name
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace("/", "__")
        local_dir = cache_root / safe_name
        if not local_dir.exists():
            snapshot_download(
                repo_id=model_name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        return str(local_dir)

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not self.available or not docs:
            return [(d, 0.0) for d in docs]
        pairs = [(query, d.page_content or "") for d in docs]
        scores = self._ce.predict(pairs, batch_size=self.batch_size)  # higher is better
        out = list(zip(docs, [float(x) for x in scores]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out


def _baseline_retrieve(
    db: FAISS,
    query: str,
    *,
    top_k: int,
    score_threshold: float,
    keyword_boost: float,
    reranker: Optional[CrossEncoderReranker],
    rerank_top_n: int,
) -> List[Document]:
    pairs = db.similarity_search_with_score(query, k=top_k)
    pairs = apply_score_threshold(pairs, score_threshold)
    if not pairs:
        return []

    docs_all = [d for d, _s in pairs]
    docs_for_rerank = docs_all[: min(rerank_top_n, len(docs_all))]

    if reranker is not None and reranker.available:
        rr = reranker.rerank(query, docs_for_rerank)  # higher is better
        rr_docs = [d for d, _score in rr]
        rest_docs = docs_all[len(docs_for_rerank):]
        return rr_docs + rest_docs

    # fallback：关键词 boost
    pairs2 = rerank_with_keyword_boost(query, pairs, boost=keyword_boost)
    return [d for d, _s in pairs2]


def _doc_key(doc: Document) -> str:
    md = doc.metadata or {}
    if md.get("id") is not None:
        return f"id:{md.get('id')}"
    source = md.get("source") or ""
    title = md.get("title") or ""
    section = md.get("section") or ""
    base = f"source:{source}|title:{title}|section:{section}"
    content = (doc.page_content or "").strip()
    if content:
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
        return f"{base}|sha1:{digest}"
    return base


def _rrf_fuse(
    query_candidates: Iterable[Tuple[QueryCandidate, List[Document]]],
    *,
    rrf_k: int,
) -> List[Document]:
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}

    for cand, ranked_docs in query_candidates:
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _doc_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + cand.weight / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[k] for k, _ in ordered]


def _retrieve_pairs_for_query(
    db: FAISS,
    query: str,
    *,
    per_query_k: int,
    score_threshold: float,
    keyword_boost: float,
) -> List[Document]:
    pairs = db.similarity_search_with_score(query, k=per_query_k)
    pairs = apply_score_threshold(pairs, score_threshold)
    if not pairs:
        return []
    pairs = rerank_with_keyword_boost(query, pairs, boost=keyword_boost)
    return [d for d, _s in pairs]


def _get_rewrite_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not cfg:
        return {}
    return (cfg.get("rag", {}) or {}).get("query_rewrite", {}) or {}


def retrieve_ranked_docs(
    db: FAISS,
    query: str,
    *,
    top_k: int,
    score_threshold: float,
    keyword_boost: float,
    reranker: Optional[CrossEncoderReranker],
    rerank_top_n: int,
    llm: Optional[BaseChatModel] = None,
    embeddings: Optional[Embeddings] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    稳定接口：给定 query，返回最终排序后的 docs（与 02_chat.py 行为一致）
    """
    rewrite_cfg = _get_rewrite_cfg(cfg)
    if not rewrite_cfg.get("enabled", False):
        return _baseline_retrieve(
            db,
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            keyword_boost=keyword_boost,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
        )

    try:
        emb = embeddings or getattr(db, "embeddings", None)
        candidates = rewrite_queries(query, llm=llm, embeddings=emb, cfg=cfg or {})
        if rewrite_cfg.get("log_rewrites", False):
            print("[query_rewrite] enabled")
            for item in candidates:
                tag = "Q0" if item.is_original else "RW"
                print(f"[query_rewrite] {tag}: {item.query}")
        if not candidates or all(c.is_original for c in candidates):
            return _baseline_retrieve(
                db,
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                keyword_boost=keyword_boost,
                reranker=reranker,
                rerank_top_n=rerank_top_n,
            )

        per_query_k = int(rewrite_cfg.get("per_query_k", 15))
        rrf_k = int(rewrite_cfg.get("rrf_k", 60))

        fused_inputs: List[Tuple[QueryCandidate, List[Document]]] = []
        for cand in candidates:
            ranked_docs = _retrieve_pairs_for_query(
                db,
                cand.query,
                per_query_k=per_query_k,
                score_threshold=score_threshold,
                keyword_boost=keyword_boost,
            )
            if ranked_docs:
                fused_inputs.append((cand, ranked_docs))

        if not fused_inputs:
            return _baseline_retrieve(
                db,
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                keyword_boost=keyword_boost,
                reranker=reranker,
                rerank_top_n=rerank_top_n,
            )

        fused_docs = _rrf_fuse(fused_inputs, rrf_k=rrf_k)
        fused_docs = fused_docs[:top_k]

        if reranker is not None and reranker.available and fused_docs:
            docs_for_rerank = fused_docs[: min(rerank_top_n, len(fused_docs))]
            rr = reranker.rerank(query, docs_for_rerank)
            rr_docs = [d for d, _score in rr]
            rest_docs = fused_docs[len(docs_for_rerank):]
            return rr_docs + rest_docs

        return fused_docs
    except Exception:
        return _baseline_retrieve(
            db,
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            keyword_boost=keyword_boost,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
        )


def _self_test() -> None:
    from src.common import load_yaml
    from src.model_factory import build_embeddings, build_llm

    cfg = load_yaml("config.yaml")
    query = os.environ.get("RAG_TEST_QUERY", "UCC: 如何调用接口生成索引")
    persist_dir = ((cfg.get("paths", {}) or {}).get("persist_dir") or "").strip()

    if not persist_dir or not os.path.isdir(persist_dir):
        print("[self-test] FAISS index not found. Please run 01_index.py first.")
        return

    embeddings = None
    llm = None
    try:
        embeddings = build_embeddings(cfg)
    except Exception:
        pass

    try:
        llm = build_llm(cfg, temperature=(cfg.get("runtime", {}) or {}).get("temperature", 0.2))
    except Exception:
        pass

    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    rewrites = rewrite_queries(query, llm=llm, embeddings=embeddings, cfg=cfg)
    print("[self-test] rewrites:")
    for item in rewrites:
        tag = "Q0" if item.is_original else "RW"
        print(f"- [{tag}] {item.query}")

    docs_ranked = retrieve_ranked_docs(
        db=db,
        query=query,
        top_k=int(((cfg.get("retrieval", {}) or {}).get("top_k") or 10)),
        score_threshold=float(((cfg.get("rag", {}) or {}).get("score_threshold") or 0)),
        keyword_boost=0.25,
        reranker=None,
        rerank_top_n=10,
        llm=llm,
        embeddings=embeddings,
        cfg=cfg,
    )
    print(f"[self-test] fused_docs={len(docs_ranked)}")


if __name__ == "__main__":
    _self_test()
