# 02_chat.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.common import load_yaml, read_text
from src.model_factory import build_embeddings, build_llm

# 把“召回→阈值→重排”的逻辑抽到独立模块里（以后改 RAG 只改那边，评测脚本不动）
from src.rag_retrieval import CrossEncoderReranker, retrieve_ranked_docs


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


def main() -> None:
    cfg = load_yaml("config.yaml")

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
    system_prompt = read_text(system_prompt_path, default="你是 UCC 文档助手。你只能基于已检索到的片段回答，禁止猜测。")

    rerank_cfg = cfg.get("rerank", {}) or {}
    rerank_enabled = bool(rerank_cfg.get("enabled", False))
    rerank_model = str(rerank_cfg.get("model", "BAAI/bge-reranker-base"))
    rerank_top_n = int(rerank_cfg.get("top_n", top_k))
    rerank_batch_size = int(rerank_cfg.get("batch_size", 16))

    print("=== RAG CHAT ===")
    print(f"- FAISS: {persist_dir}")
    print(f"- embed: {embed_model}")
    print(f"- llm:   {llm_model}")
    print(f"- top_k: {top_k}")
    print(f"- show_k: {show_k}")
    print(f"- max_ctx: {max_ctx}")
    print(f"- score_threshold: {score_threshold} (<=0 means disabled)")
    print(f"- keyword_boost: {keyword_boost}")
    print(f"- system_prompt: {system_prompt_path}")

    reranker: Optional[CrossEncoderReranker] = None
    if rerank_enabled:
        reranker = CrossEncoderReranker(rerank_model, batch_size=rerank_batch_size)
        if reranker.available:
            print(f"[rerank] enabled model={rerank_model} batch_size={rerank_batch_size}")
        else:
            print(f"[rerank] init failed -> fallback to keyword boost. err={reranker._init_error!r}")
            reranker = None

    print("输入问题，输入 exit 退出。\n")

    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"FAISS index dir not found: {persist_dir}\nPlease run 01_index.py first.")

    embeddings = build_embeddings(cfg)
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    llm = build_llm(cfg, temperature=temperature)

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        # 统一入口：召回→阈值→重排（以后改 RAG 流程只改 src.rag_retrieval）
        docs_ranked = retrieve_ranked_docs(
            db=db,
            query=q,
            top_k=top_k,
            score_threshold=score_threshold,
            keyword_boost=keyword_boost,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
        )

        if not docs_ranked:
            print("\n--- Retrieved ---")
            print("(no hits)")
            print("--- End ---\n")
            print("A>")
            print("文档中未找到相关信息\n")
            continue

        # 打印命中
        print("\n--- Retrieved ---")
        for i, d in enumerate(docs_ranked[:show_k], start=1):
            m = safe_meta(d)
            src = m.get("source", "?")
            did = m.get("id", "?")
            title = m.get("title", "")
            section = m.get("section", "")
            print(f"[{i}] {src} | {did} | {title} | {section}")
            snip = (d.page_content or "").strip().replace("\n", " ")
            print("   ", snip[:240] + ("..." if len(snip) > 240 else ""))
        print("--- End ---\n")

        # 喂给 LLM
        docs_for_llm = docs_ranked[:max_ctx]
        context_block = build_context_block(docs_for_llm)
        messages = build_messages(system_prompt, q, context_block)

        resp = llm.invoke(messages)
        ans = getattr(resp, "content", str(resp))
        print("A>")
        print((ans or "").strip())
        print()


if __name__ == "__main__":
    main()
