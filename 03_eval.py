# 03_eval.py
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.common import load_yaml
from src.rag_retrieval import CrossEncoderReranker, retrieve_ranked_docs
from src.model_factory import build_embeddings, build_llm


# =========================
# IO helpers
# =========================
def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if s:
                rows.append(json.loads(s))
    return rows


def dump_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_doc_id(doc: Document) -> str:
    md = doc.metadata or {}
    return str(md.get("id") or "").strip()


# =========================
# Metrics
# =========================
def precision_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = pred[:k]
    hit = sum(1 for x in top if x in rel)
    return hit / k


def recall_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    if not rel:
        return 0.0
    top = pred[:k]
    hit = sum(1 for x in top if x in rel)
    return hit / len(rel)


def f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def mrr_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    for i, x in enumerate(pred[:k], start=1):
        if x in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred: List[str], rel: Set[str], k: int) -> float:
    # binary relevance
    def dcg(items: List[str]) -> float:
        s = 0.0
        for i, x in enumerate(items, start=1):
            gain = 1.0 if x in rel else 0.0
            s += gain / math.log2(i + 1)
        return s

    top = pred[:k]
    dcg_val = dcg(top)

    # ideal DCG: k 个里全部 relevant（binary）时的最优
    ideal_hits = min(len(rel), k)
    if ideal_hits <= 0:
        return 0.0
    ideal = ["__REL__"] * ideal_hits
    # 让 dcg() 认为全 relevant
    def dcg_ideal(n: int) -> float:
        s = 0.0
        for i in range(1, n + 1):
            s += 1.0 / math.log2(i + 1)
        return s

    idcg_val = dcg_ideal(ideal_hits)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# =========================
# Diagnostic helpers
# =========================
COMMON_TYPO_MAP = {
    "mosbus": "modbus",
    "sqllite": "sqlite",
    "excle": "excel",
    "jason": "json",
    "httpp": "http",
    "my sql": "mysql",
}


def normalize_query(q: str) -> str:
    s = (q or "").strip().lower()
    s = " ".join(s.split())
    for k, v in COMMON_TYPO_MAP.items():
        s = s.replace(k, v)
    return s


def classify_query(q: str) -> str:
    """
    仅用于诊断分组，不影响检索
    """
    raw = q or ""
    s = normalize_query(raw)

    # 多段复杂：两段或以上
    if "\n\n" in raw or raw.count("\n") >= 2:
        return "complex_multiparagraph"

    # 过短口语：很短
    if len(s) <= 6:
        return "short_utterance"

    # 中英混合
    has_ascii = any("a" <= c <= "z" for c in s)
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in raw)
    if has_ascii and has_cjk:
        return "mixed_zh_en"
    if has_ascii and not has_cjk:
        return "english_only"

    # 错别字（启发式：原始 query 包含 typo key）
    raw_low = raw.lower()
    if any(k in raw_low for k in COMMON_TYPO_MAP.keys()):
        return "typo_like"

    # 口语化/改写（启发式）
    if any(w in raw for w in ["怎么", "如何", "能否", "能不能", "示例", "例子", "怎么写", "怎么做"]):
        return "paraphrase_like"

    return "standard"


def first_hit_rank(pred: List[str], rel: Set[str], k: int) -> int:
    for i, x in enumerate(pred[:k], start=1):
        if x in rel:
            return i
    return 0


@dataclass
class PerQueryResult:
    qid: str
    qtype: str
    query: str
    relevant_ids: List[str]
    pred_topk: List[str]
    hit_count: int
    first_rank: int
    p: float
    r: float
    mrr: float
    ndcg: float


def group_summary(results: List[PerQueryResult], k: int) -> Dict[str, dict]:
    groups: Dict[str, List[PerQueryResult]] = {}
    for r in results:
        groups.setdefault(r.qtype, []).append(r)

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    out: Dict[str, dict] = {}
    for g, items in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        out[g] = {
            "count": len(items),
            f"P@{k}": avg([x.p for x in items]),
            f"R@{k}": avg([x.r for x in items]),
            f"MRR@{k}": avg([x.mrr for x in items]),
            f"nDCG@{k}": avg([x.ndcg for x in items]),
            # 没命中的用 k+1 计入平均，能直观看“平均要翻到多后面才能见到 relevant”
            f"avg_first_hit_rank@{k}": avg([x.first_rank if x.first_rank > 0 else (k + 1) for x in items]),
            f"miss_rate@{k}": sum(1 for x in items if x.first_rank == 0) / len(items),
        }
    return out


# =========================
# Main
# =========================
def main() -> None:
    cfg = load_yaml("config.yaml")

    queries_path = os.path.join("data", "eval_queries.jsonl")
    qrels_path = os.path.join("data", "eval_qrels.jsonl")

    persist_dir = cfg.get("paths", {}).get("persist_dir", "index/faiss_ucc")
    embed_model = cfg.get("models", {}).get("embed_model", "nomic-embed-text:latest")

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    top_k = int(retrieval_cfg.get("top_k", 50))
    show_k = int(retrieval_cfg.get("show_k", 10))
    k_eval = show_k  # 评测用前 show_k（通常 10）

    runtime_cfg = cfg.get("runtime", {}) or {}
    temperature = float(cfg.get("generation", {}).get("temperature", runtime_cfg.get("temperature", 0.2)))

    rag_cfg = cfg.get("rag", {}) or {}
    score_threshold = float(rag_cfg.get("score_threshold", 0.0))
    keyword_boost = float(rag_cfg.get("keyword_boost", 0.25))

    rerank_cfg = cfg.get("rerank", {}) or {}
    rerank_enabled = bool(rerank_cfg.get("enabled", False))
    rerank_model = str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))
    rerank_top_n = int(rerank_cfg.get("top_n", top_k))
    rerank_batch_size = int(rerank_cfg.get("batch_size", 16))
    rerank_cache_dir = str(rerank_cfg.get("cache_dir", "models/rerank"))

    # ---- load eval data ----
    if not Path(queries_path).exists():
        raise FileNotFoundError(f"eval_queries not found: {queries_path}")
    if not Path(qrels_path).exists():
        raise FileNotFoundError(f"eval_qrels not found: {qrels_path}")

    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)
    rel_map: Dict[str, Set[str]] = {row["qid"]: set(row.get("relevant_ids", [])) for row in qrels}

    # ---- load FAISS ----
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"FAISS index dir not found: {persist_dir}")

    embeddings = build_embeddings(cfg)
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    llm = build_llm(cfg, temperature=temperature)

    # 加在 load FAISS 后面（db = FAISS.load_local(...) 之后）
    print("\n[index sanity] sample doc ids:")
    docs = db.docstore._dict  # type: ignore
    cnt = 0
    for _k, d in docs.items():
        mid = (d.metadata or {}).get("id")
        src = (d.metadata or {}).get("source")
        if mid:
            print(" -", mid, "|", src)
            cnt += 1
        if cnt >= 10:
            break

    # 统计 Demo 覆盖率
    demo_cnt = 0
    for _k, d in docs.items():
        mid = str((d.metadata or {}).get("id") or "")
        if mid.startswith("Demo_"):
            demo_cnt += 1
    print(f"[index sanity] demo_docs={demo_cnt} total_docs={len(docs)}")

    # ---- reranker（与 02_chat 行为一致）----
    reranker = None
    if rerank_enabled:
        rr = CrossEncoderReranker(
            rerank_model,
            batch_size=rerank_batch_size,
            cache_dir=rerank_cache_dir,
        )
        if getattr(rr, "available", False):
            reranker = rr
        else:
            err = getattr(rr, "_init_error", None)
            print(f"[rerank] init failed -> fallback keyword boost. err={err!r}")

    # ---- eval loop ----
    ps, rs, f1s, mrrs, ndcgs = [], [], [], [], []
    empty_cnt = 0
    skipped = 0

    per_results: List[PerQueryResult] = []

    for row in queries:
        qid = row["qid"]
        query = row["query"]
        qtype = classify_query(query)

        rel = rel_map.get(qid)
        if rel is None:
            skipped += 1
            continue

        docs_ranked = retrieve_ranked_docs(
            db=db,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            keyword_boost=keyword_boost,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
            llm=llm,
            embeddings=embeddings,
            cfg=cfg,
        )

        if not docs_ranked:
            empty_cnt += 1
            pred_ids: List[str] = []
        else:
            pred_ids = [safe_doc_id(d) for d in docs_ranked]
            pred_ids = [x for x in pred_ids if x]

        p = precision_at_k(pred_ids, rel, k_eval)
        r = recall_at_k(pred_ids, rel, k_eval)
        mrr = mrr_at_k(pred_ids, rel, k_eval)
        ndcg = ndcg_at_k(pred_ids, rel, k_eval)
        fr = first_hit_rank(pred_ids, rel, k_eval)
        hit_cnt = sum(1 for x in pred_ids[:k_eval] if x in rel)

        ps.append(p)
        rs.append(r)
        f1s.append(f1(p, r))
        mrrs.append(mrr)
        ndcgs.append(ndcg)

        per_results.append(
            PerQueryResult(
                qid=qid,
                qtype=qtype,
                query=query,
                relevant_ids=sorted(list(rel)),
                pred_topk=pred_ids[:k_eval],
                hit_count=hit_cnt,
                first_rank=fr,
                p=p,
                r=r,
                mrr=mrr,
                ndcg=ndcg,
            )
        )

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    n = len(ps)
    empty_rate = empty_cnt / n if n else 0.0

    # ---- overall report ----
    print("\n=== OFFLINE RETRIEVAL EVAL ===")
    print(f"queries_used: {n}")
    print(f"queries_skipped(no qrels): {skipped}")
    print(f"empty_rate: {empty_rate:.4f}")
    print(f"top_k(retrieve): {top_k}")
    print(f"k_eval: {k_eval}")
    print(f"P@{k_eval}:   {avg(ps):.4f}")
    print(f"R@{k_eval}:   {avg(rs):.4f}")
    print(f"F1@{k_eval}:  {avg(f1s):.4f}")
    print(f"MRR@{k_eval}: {avg(mrrs):.4f}")
    print(f"nDCG@{k_eval}:{avg(ndcgs):.4f}")

    # ---- diagnostics outputs ----
    report_path = os.path.join("data", "eval_report.jsonl")
    dump_jsonl(
        report_path,
        [
            {
                "qid": x.qid,
                "qtype": x.qtype,
                "query": x.query,
                "relevant_ids": x.relevant_ids,
                f"pred_top{k_eval}": x.pred_topk,
                "hit_count": x.hit_count,
                "first_hit_rank": x.first_rank,
                f"P@{k_eval}": x.p,
                f"R@{k_eval}": x.r,
                f"MRR@{k_eval}": x.mrr,
                f"nDCG@{k_eval}": x.ndcg,
            }
            for x in per_results
        ],
    )
    print(f"\n[diagnostic] per-query report saved -> {report_path}")

    misses = [x for x in per_results if x.first_rank == 0]
    print(f"\n[diagnostic] misses@{k_eval}: {len(misses)}/{len(per_results)}")
    for x in misses[:15]:
        q_short = (x.query or "").replace("\n", " ")
        if len(q_short) > 90:
            q_short = q_short[:90] + "..."
        print(f"- {x.qid} [{x.qtype}] Q={q_short!r} rel={x.relevant_ids[:6]}")

    gs = group_summary(per_results, k_eval)
    print("\n=== GROUPED SUMMARY ===")
    for g, s in gs.items():
        print(
            f"{g:24s} n={s['count']:>3d} "
            f"P@{k_eval}={s[f'P@{k_eval}']:.4f} "
            f"R@{k_eval}={s[f'R@{k_eval}']:.4f} "
            f"MRR@{k_eval}={s[f'MRR@{k_eval}']:.4f} "
            f"nDCG@{k_eval}={s[f'nDCG@{k_eval}']:.4f} "
            f"miss={s[f'miss_rate@{k_eval}']:.2%} "
            f"avg_first_rank={s[f'avg_first_hit_rank@{k_eval}']:.2f}"
        )


if __name__ == "__main__":
    main()
