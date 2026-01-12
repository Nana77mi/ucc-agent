from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass(frozen=True)
class QueryCandidate:
    query: str
    weight: float
    is_original: bool


_REWRITE_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_:+-]+", (text or ""))


def _token_coverage_ratio(base_tokens: List[str], candidate: str) -> float:
    if not base_tokens:
        return 1.0
    cand = " ".join(_extract_tokens(candidate)).lower()
    hit = sum(1 for t in base_tokens if t.lower() in cand)
    return hit / max(len(base_tokens), 1)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _parse_json_array(text: str) -> List[Any]:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return []


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = _normalize_query(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _build_prompt(query: str, *, num_rewrites: int, max_len: int, colon_tokens: List[str]) -> List[Any]:
    colon_hint = "无" if not colon_tokens else ", ".join(colon_tokens)
    system = (
        "你是搜索查询改写助手。"
        "根据用户问题生成多个更精准或等价的检索查询。"
        "必须输出 JSON 数组，每个元素为 {\"q\": <string>}。"
        "不要输出任何解释或额外文字。"
        "保持原始问题的关键专有名词；如果包含类似 'X:' 前缀，必须保留。"
        "每条查询必须紧凑，不要偏题。"
    )
    user = (
        f"原始问题: {query}\n"
        f"需要生成 {num_rewrites} 条不同的改写（不包含原始问题）。\n"
        f"最大长度: {max_len} 字符。\n"
        f"需要保留的前缀: {colon_hint}.\n"
        "仅输出 JSON 数组。"
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _extract_rewrite_strings(parsed: List[Any]) -> List[str]:
    out: List[str] = []
    for item in parsed:
        if isinstance(item, dict) and "q" in item:
            q = str(item.get("q") or "").strip()
            if q:
                out.append(q)
            continue
        if isinstance(item, str):
            q = item.strip()
            if q:
                out.append(q)
    return out


def _get_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("rag", {}) or {}).get("query_rewrite", {}) or {}


def rewrite_queries(
    query: str,
    *,
    llm: Optional[BaseChatModel],
    embeddings: Optional[Embeddings],
    cfg: Dict[str, Any],
) -> List[QueryCandidate]:
    rewrite_cfg = _get_cfg(cfg)
    enabled = bool(rewrite_cfg.get("enabled", False))
    num_rewrites = int(rewrite_cfg.get("num_rewrites", 4))
    max_len = int(rewrite_cfg.get("max_query_len", 64))
    sim_threshold = float(rewrite_cfg.get("sim_threshold", 0.8))
    require_colon = bool(rewrite_cfg.get("require_colon_prefix_preserve", True))
    cache_ttl = int(rewrite_cfg.get("cache_ttl_sec", 86400))
    weights = rewrite_cfg.get("weights", {}) or {}
    weight_q0 = float(weights.get("q0", 1.0))
    weight_rw = float(weights.get("rewrite", 0.85))

    q0 = (query or "").strip()
    if not q0:
        return []

    if not enabled:
        return [QueryCandidate(query=q0, weight=weight_q0, is_original=True)]

    cache_key = _normalize_query(q0)
    now = time.time()
    cached = _REWRITE_CACHE.get(cache_key)
    if cached and now - cached.get("ts", 0) <= cache_ttl:
        cached_queries = cached.get("items", [])
        if isinstance(cached_queries, list) and cached_queries:
            return _build_candidates(q0, cached_queries, weight_q0, weight_rw)

    rewrites: List[str] = []
    if llm is not None and num_rewrites > 0:
        colon_tokens = [t for t in _extract_tokens(q0) if ":" in t]
        messages = _build_prompt(q0, num_rewrites=num_rewrites, max_len=max_len, colon_tokens=colon_tokens)
        try:
            resp = llm.invoke(messages)
            text = getattr(resp, "content", str(resp))
            parsed = _parse_json_array(text)
            rewrites = _extract_rewrite_strings(parsed)
        except Exception:
            rewrites = []

    rewrites = [q[:max_len].strip() for q in rewrites if q and q.strip()]
    rewrites = _dedupe_keep_order(rewrites)

    gated = _gate_rewrites(
        q0,
        rewrites,
        embeddings=embeddings,
        sim_threshold=sim_threshold,
        require_colon=require_colon,
    )

    gated = gated[:num_rewrites]
    all_queries = [q0] + gated

    _REWRITE_CACHE[cache_key] = {"ts": now, "items": all_queries}
    return _build_candidates(q0, all_queries, weight_q0, weight_rw)


def _build_candidates(q0: str, queries: List[str], weight_q0: float, weight_rw: float) -> List[QueryCandidate]:
    out: List[QueryCandidate] = []
    for q in _dedupe_keep_order(queries):
        is_original = _normalize_query(q) == _normalize_query(q0)
        weight = weight_q0 if is_original else weight_rw
        out.append(QueryCandidate(query=q, weight=weight, is_original=is_original))
    if not out:
        out.append(QueryCandidate(query=q0, weight=weight_q0, is_original=True))
    return out


def _gate_rewrites(
    q0: str,
    candidates: List[str],
    *,
    embeddings: Optional[Embeddings],
    sim_threshold: float,
    require_colon: bool,
) -> List[str]:
    if not candidates:
        return []

    base_tokens = _extract_tokens(q0)
    colon_tokens = [t for t in base_tokens if ":" in t]

    q0_vec: Optional[List[float]] = None
    if embeddings is not None:
        try:
            q0_vec = embeddings.embed_query(q0)
        except Exception:
            q0_vec = None

    filtered: List[str] = []
    for cand in candidates:
        if require_colon and colon_tokens:
            if not all(t in cand for t in colon_tokens):
                continue

        coverage = _token_coverage_ratio(base_tokens, cand)
        if coverage < 0.5:
            continue

        if q0_vec is not None:
            try:
                cand_vec = embeddings.embed_query(cand)
                sim = _cosine_sim(q0_vec, cand_vec)
                if sim < sim_threshold:
                    continue
            except Exception:
                pass

        filtered.append(cand)

    return filtered


if __name__ == "__main__":
    from src.common import load_yaml
    from src.model_factory import build_embeddings, build_llm

    cfg = load_yaml("config.yaml")
    query = "UCC: 如何调用接口生成索引"
    try:
        llm = build_llm(cfg, temperature=(cfg.get("runtime", {}) or {}).get("temperature", 0.2))
    except Exception:
        llm = None
    try:
        embeddings = build_embeddings(cfg)
    except Exception:
        embeddings = None

    candidates = rewrite_queries(query, llm=llm, embeddings=embeddings, cfg=cfg)
    print("rewrites:")
    for item in candidates:
        tag = "Q0" if item.is_original else "RW"
        print(f"- [{tag}] {item.query} (weight={item.weight:.2f})")
