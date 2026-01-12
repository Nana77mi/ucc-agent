# 04_stability.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.common import load_yaml, read_text
from src.model_factory import build_embeddings, build_llm
from src.rag_retrieval import CrossEncoderReranker, retrieve_ranked_docs


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


def safe_meta(doc: Document) -> Dict[str, str]:
    return doc.metadata or {}


def safe_doc_id(doc: Document) -> str:
    md = doc.metadata or {}
    return str(md.get("id") or "").strip()


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


def build_messages(system_prompt: str, user_query: str, context_block: str) -> List[dict]:
    sys = (system_prompt or "").strip()
    user = f"""你是文档问答助手。

请严格基于下方 Context 回答问题，并在回答中用 [1][2] 的形式标注引用来源。
如果无法从 Context 得到答案，请回答“文档中未找到相关信息”。

【Context】
{context_block}

【Question】
{user_query}

【Answer】
"""
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    if denom <= 0:
        return 0.0
    return dot / denom


def pairwise_similarity(vectors: List[List[float]]) -> Dict[str, float]:
    if len(vectors) < 2:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    sims: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_similarity(vectors[i], vectors[j]))
    if not sims:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": sum(sims) / len(sims), "min": min(sims), "max": max(sims)}


def run_llm_with_retries(
    llm,
    messages: List[dict],
    *,
    max_retries: int,
    retry_backoff: float,
) -> tuple[str, Optional[Exception]]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = llm.invoke(messages)
            ans = getattr(resp, "content", str(resp))
            return (ans or "").strip(), None
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_backoff)
    return "", last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG/LLM stability on repeated runs.")
    parser.add_argument("--runs", type=int, default=3, help="Repeat count per query.")
    parser.add_argument("--queries", type=str, default=os.path.join("data", "eval_queries.jsonl"))
    parser.add_argument("--report", type=str, default=os.path.join("data", "stability_report.jsonl"))
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        help="Sample count for representative queries (0 means all).",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="LLM retry attempts per run.")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Backoff seconds between retries.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation and answer similarity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml("config.yaml")

    persist_dir = cfg.get("paths", {}).get("persist_dir", "index/faiss_ucc")
    system_prompt_path = cfg.get("paths", {}).get("system_prompt", "prompts/system.txt")
    system_prompt = read_text(system_prompt_path, default="你是 UCC 文档助手。你只能基于已检索到的片段回答，禁止猜测。")

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    top_k = int(retrieval_cfg.get("top_k", 50))
    show_k = int(retrieval_cfg.get("show_k", 10))

    runtime_cfg = cfg.get("runtime", {}) or {}
    max_ctx = int(runtime_cfg.get("max_context_docs", 10))
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

    if not Path(args.queries).exists():
        raise FileNotFoundError(f"queries not found: {args.queries}")
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"FAISS index dir not found: {persist_dir}")

    queries = read_jsonl(args.queries)
    if args.sample_size and args.sample_size > 0 and len(queries) > args.sample_size:
        step = max(len(queries) // args.sample_size, 1)
        sampled = [queries[i] for i in range(0, len(queries), step)][: args.sample_size]
        queries = sampled
    embeddings = build_embeddings(cfg)
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    llm = None if args.skip_llm else build_llm(cfg, temperature=temperature)

    reranker: Optional[CrossEncoderReranker] = None
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

    print("\n=== STABILITY CHECK ===")
    print(f"queries: {len(queries)}")
    print(f"sample_size: {args.sample_size}")
    print(f"runs_per_query: {args.runs}")
    print(f"top_k(retrieve): {top_k} | compare_top_k: {show_k}")
    print(f"temperature: {temperature}")
    if args.skip_llm:
        print("llm: skipped")

    report_rows: List[dict] = []
    rag_consistent_cnt = 0
    sim_avgs: List[float] = []

    for row in queries:
        qid = row.get("qid", "")
        query = row.get("query", "")
        if not query:
            continue

        rag_lists: List[List[str]] = []
        answers: List[str] = []
        llm_failures = 0
        llm_total = 0

        for _i in range(args.runs):
            docs_ranked: List[Document]
            try:
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
            except Exception:
                rag_lists.append([])
                if llm is not None:
                    llm_total += 1
                    llm_failures += 1
                    answers.append("")
                continue

            doc_ids = [safe_doc_id(d) for d in docs_ranked[:show_k] if safe_doc_id(d)]
            rag_lists.append(doc_ids)

            if llm is None:
                continue

            docs_for_llm = docs_ranked[:max_ctx]
            context_block = build_context_block(docs_for_llm)
            messages = build_messages(system_prompt, query, context_block)
            llm_total += 1

            try:
                ans, last_error = run_llm_with_retries(
                    llm,
                    messages,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                answers.append(ans)
            except Exception as exc:
                last_error = exc
                answers.append("")

            if last_error is not None:
                llm_failures += 1

        base = rag_lists[0] if rag_lists else []
        rag_consistent = all(x == base for x in rag_lists)
        rag_match_rate = sum(1 for x in rag_lists if x == base) / len(rag_lists) if rag_lists else 0.0

        valid_answers = [a for a in answers if a]
        ans_vectors = embeddings.embed_documents(valid_answers) if len(valid_answers) >= 2 else []
        sim_stats = pairwise_similarity(ans_vectors)

        rag_consistent_cnt += 1 if rag_consistent else 0
        sim_avgs.append(sim_stats["avg"])

        report_rows.append(
            {
                "qid": qid,
                "query": query,
                "runs": args.runs,
                "rag_consistent": rag_consistent,
                "rag_match_rate": rag_match_rate,
                "rag_topk": base,
                "answer_similarity_avg": sim_stats["avg"],
                "answer_similarity_min": sim_stats["min"],
                "answer_similarity_max": sim_stats["max"],
                "answer_total": len(answers),
                "answer_valid": len(valid_answers),
                "llm_failures": llm_failures,
                "llm_total": llm_total,
            }
        )

    dump_jsonl(args.report, report_rows)
    total = len(report_rows)
    rag_consistency_rate = rag_consistent_cnt / total if total else 0.0
    sim_avg_overall = sum(sim_avgs) / len(sim_avgs) if sim_avgs else 0.0

    print(f"rag_consistency_rate: {rag_consistency_rate:.4f}")
    print(f"answer_similarity_avg: {sim_avg_overall:.4f}")
    print(f"report saved: {args.report}")


if __name__ == "__main__":
    main()
