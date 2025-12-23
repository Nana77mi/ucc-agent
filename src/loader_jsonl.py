# src/loader_jsonl.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document


def extract_content(obj: Dict[str, Any]) -> str:
    """
    尽量少处理：只提取正文内容。
    兼容 ucc_rag.jsonl 常见字段：ucc / text / content / page_content
    """
    for k in ("ucc", "text", "content", "page_content"):
        v = obj.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def load_jsonl_as_documents(
    jsonl_path: str,
    *,
    chunk_size: int = 0,   # 兼容旧接口，但不再使用
    overlap: int = 0,      # 兼容旧接口，但不再使用
    verbose_warn_empty: bool = True,
) -> List[Document]:
    """
    重要：每条 jsonl 记录只生成 1 个 Document（不再切 chunk）。
    """
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path} (cwd={Path.cwd()})")

    docs: List[Document] = []
    empty_cnt = 0

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 跳过坏行
                continue

            content = extract_content(obj)
            if not content:
                empty_cnt += 1
                if verbose_warn_empty and empty_cnt <= 20:
                    print(
                        f"[WARN] empty content: file={jsonl_path} line={line_no} "
                        f"id={obj.get('id')} source={obj.get('source')} keys={list(obj.keys())}"
                    )
                continue

            meta: Dict[str, Any] = {
                "id": obj.get("id"),
                "source": obj.get("source"),
                "kind": obj.get("kind"),
                "trigger": obj.get("trigger"),
                "tags": obj.get("tags", []),
                "_file": str(p.as_posix()),
                "_line": line_no,
                # 不再切块：统一认为 chunk_id=1, chunk_total=1（方便 02_chat 展示逻辑不改）
                "chunk_id": 1,
                "chunk_total": 1,
            }

            # 保留常用字段（少处理）
            for k in ("path", "title", "section", "anchor"):
                if k in obj and obj.get(k) is not None:
                    meta[k] = obj.get(k)

            docs.append(Document(page_content=content, metadata=meta))

    if verbose_warn_empty and empty_cnt > 20:
        print(f"[WARN] empty content lines: file={jsonl_path} count={empty_cnt} (only first 20 shown)")

    return docs


def load_many(
    paths: List[str],
    *,
    chunk_size: int = 0,   # 兼容旧接口，但不再使用
    overlap: int = 0,      # 兼容旧接口，但不再使用
) -> List[Document]:
    """
    多文件加载。每条 json 记录 -> 1 doc。
    仍打印每个文件贡献的 doc 数，便于调试。
    """
    all_docs: List[Document] = []

    for p in paths:
        if p.lower().endswith(".jsonl"):
            one = load_jsonl_as_documents(p, chunk_size=chunk_size, overlap=overlap)
            print(f"[LOAD] {p} -> {len(one)} docs")
            all_docs.extend(one)
        else:
            from langchain_community.document_loaders import TextLoader

            one = TextLoader(p, encoding="utf-8").load()
            print(f"[LOAD] {p} -> {len(one)} docs")
            all_docs.extend(one)

    return all_docs
