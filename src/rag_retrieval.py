# src/rag_retrieval.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

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
    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.available = False
        self._init_error: Optional[Exception] = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._ce = CrossEncoder(model_name)
            self.available = True
        except Exception as e:
            self._ce = None
            self.available = False
            self._init_error = e

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not self.available or not docs:
            return [(d, 0.0) for d in docs]
        pairs = [(query, d.page_content or "") for d in docs]
        scores = self._ce.predict(pairs, batch_size=self.batch_size)  # higher is better
        out = list(zip(docs, [float(x) for x in scores]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

def retrieve_ranked_docs(
    db: FAISS,
    query: str,
    *,
    top_k: int,
    score_threshold: float,
    keyword_boost: float,
    reranker: Optional[CrossEncoderReranker],
    rerank_top_n: int,
) -> List[Document]:
    """
    稳定接口：给定 query，返回最终排序后的 docs（与 02_chat.py 行为一致）
    """
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
