# 02_chat.py
from __future__ import annotations

from typing import List

from src.chat_engine import ChatEngine, Message, safe_meta


def main() -> None:
    engine = ChatEngine.from_yaml("config.yaml")
    settings = engine.settings

    print("=== RAG CHAT ===")
    print(f"- FAISS: {settings.persist_dir}")
    print(f"- embed: {settings.embed_model}")
    print(f"- llm:   {settings.llm_model}")
    print(f"- top_k: {settings.top_k}")
    print(f"- show_k: {settings.show_k}")
    print(f"- max_ctx: {settings.max_ctx}")
    print(f"- score_threshold: {settings.score_threshold} (<=0 means disabled)")
    print(f"- keyword_boost: {settings.keyword_boost}")
    print(f"- system_prompt: {settings.system_prompt_path}")
    if engine.reranker_status():
        print("[rerank] enabled")
    else:
        print("[rerank] disabled -> fallback to keyword boost")

    print("输入问题，输入 exit 退出。\n")

    history: List[Message] = []

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        response = engine.answer(q, history=history)
        docs_ranked = response.docs

        if not docs_ranked:
            print("\n--- Retrieved ---")
            print("(no hits)")
            print("--- End ---\n")
            print("A>")
            print("文档中未找到相关信息\n")
            history.extend([
                {"role": "user", "content": q},
                {"role": "assistant", "content": "文档中未找到相关信息"},
            ])
            continue

        print("\n--- Retrieved ---")
        for i, d in enumerate(docs_ranked[: settings.show_k], start=1):
            m = safe_meta(d)
            src = m.get("source", "?")
            did = m.get("id", "?")
            title = m.get("title", "")
            section = m.get("section", "")
            print(f"[{i}] {src} | {did} | {title} | {section}")
            snip = (d.page_content or "").strip().replace("\n", " ")
            print("   ", snip[:240] + ("..." if len(snip) > 240 else ""))
        print("--- End ---\n")

        print("A>")
        print((response.answer or "").strip())
        metrics = response.metrics
        if not metrics or metrics.tokens_per_s is None:
            elapsed = metrics.elapsed if metrics else 0.0
            print(f"[metrics] duration: {elapsed:.2f}s | tokens/s: N/A")
        else:
            print(f"[metrics] duration: {metrics.elapsed:.2f}s | tokens/s: {metrics.tokens_per_s:.2f}")
        print()

        history.extend([
            {"role": "user", "content": q},
            {"role": "assistant", "content": response.answer},
        ])


if __name__ == "__main__":
    main()
