# 01_index.py
from __future__ import annotations

import os
import shutil
import time
from typing import Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.common import load_yaml
from src.loader_jsonl import load_many


def batched(lst: List[Document], n: int) -> Iterable[List[Document]]:
    n = max(1, int(n))
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def ensure_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main() -> None:
    cfg = load_yaml("config.yaml")

    persist_dir = cfg.get("paths", {}).get("persist_dir", "index/faiss_ucc")
    input_jsonl = cfg.get("paths", {}).get("input_jsonl", [])

    if isinstance(input_jsonl, str):
        input_paths = [input_jsonl]
    else:
        input_paths = list(input_jsonl or [])

    embed_model = cfg.get("models", {}).get("embed_model", "nomic-embed-text:latest")
    embeddings = OllamaEmbeddings(model=embed_model)

    index_cfg = cfg.get("index", {}) or {}
    rebuild = bool(index_cfg.get("rebuild", True))
    batch_size = int(index_cfg.get("batch_size", 64))
    sleep_between = float(index_cfg.get("sleep_between", 0.0))

    print("=== Build FAISS Index ===")
    print("CWD =", os.getcwd())
    print("persist_dir =", persist_dir)
    print("embed_model =", embed_model)
    print("chunking = per-line jsonl (forced)")  # ✅ 强制逐行
    print(f"rebuild={rebuild} batch_size={batch_size} sleep_between={sleep_between}")
    print("inputs:")
    for p in input_paths:
        print(" -", p, "exists=", os.path.exists(p), "abs=", os.path.abspath(p))

    if not input_paths:
        raise SystemExit("config.yaml: paths.input_jsonl is empty")

    if rebuild:
        ensure_empty_dir(persist_dir)
    else:
        os.makedirs(persist_dir, exist_ok=True)

    # ✅ 每行一个 Document（loader 内部已强制）
    docs = load_many(input_paths)
    print(f"Loaded docs total: {len(docs)}")
    if not docs:
        raise SystemExit("No documents loaded. Check your jsonl paths and file contents.")

    db: Optional[FAISS] = None
    skipped = 0
    bad_log_path = os.path.join(persist_dir, "skipped_docs.log")

    def log_skip(d: Document, err: Exception) -> None:
        nonlocal skipped
        skipped += 1
        m = d.metadata or {}
        msg = (
            f"SKIP len={len(d.page_content)} "
            f"file={m.get('_file')} line={m.get('_line')} "
            f"err={repr(err)}\n"
        )
        print(msg.strip())
        with open(bad_log_path, "a", encoding="utf-8") as wf:
            wf.write(msg)

    for bi, batch_docs in enumerate(batched(docs, batch_size), start=1):
        try:
            if db is None:
                db = FAISS.from_documents(batch_docs, embeddings)
            else:
                db.add_documents(batch_docs)

        except Exception as e:
            # 批处理失败：逐条尝试，坏的跳过
            for d in batch_docs:
                try:
                    if db is None:
                        db = FAISS.from_documents([d], embeddings)
                    else:
                        db.add_documents([d])
                except Exception as e2:
                    log_skip(d, e2)

        if bi % 10 == 0:
            approx = min(bi * batch_size, len(docs))
            print(f"Embedded batches: {bi} ({approx}/{len(docs)}) skipped={skipped}")

        if sleep_between > 0:
            time.sleep(sleep_between)

    assert db is not None
    db.save_local(persist_dir)
    print(f"Saved FAISS index to: {persist_dir} | skipped={skipped}")
    print(f"Skipped log: {bad_log_path}")


if __name__ == "__main__":
    main()
