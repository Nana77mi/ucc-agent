# src/loader_jsonl.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


def _pick_first(d: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def _normalize_tags(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    # 允许 "a,b,c" / "a; b; c"
    s = str(v).strip()
    if not s:
        return []
    if "," in s:
        parts = s.split(",")
    elif ";" in s:
        parts = s.split(";")
    else:
        parts = [s]
    return [p.strip() for p in parts if p.strip()]


def _build_page_content(obj: Dict[str, Any]) -> str:
    """
    把一条方法/示例/文档条目拼成更适合 embedding 的文本。
    你可以按需要再扩展字段，但不要依赖具体 schema。
    """
    lines: List[str] = []

    fid = _pick_first(obj, ["id", "uid", "key"], "")
    name = _pick_first(obj, ["name", "title", "func", "method"], "")
    desc = _pick_first(obj, ["command", "desc", "description", "brief"], "")
    grammar = _pick_first(obj, ["grammar", "syntax", "sig", "signature"], "")
    signatures = obj.get("signatures")

    if fid:
        lines.append(f"ID: {fid}")
    if name:
        lines.append(f"NAME: {name}")
    if grammar:
        lines.append(f"SYNTAX: {grammar}")
    if isinstance(signatures, list) and signatures:
        lines.append("SIGNATURES:")
        for sig in signatures[:20]:
            ss = str(sig).strip()
            if ss:
                lines.append(f"- {ss}")
    elif isinstance(signatures, str) and signatures.strip():
        lines.append("SIGNATURE:\n" + signatures.strip())
    if desc:
        lines.append(f"DESC: {desc}")

    tags = _normalize_tags(obj.get("tags"))
    if tags:
        lines.append("TAGS: " + ", ".join(tags))

    # 常见字段：examples / demo / example
    ex = obj.get("examples") or obj.get("example") or obj.get("demo")
    if isinstance(ex, list) and ex:
        lines.append("EXAMPLES:")
        for x in ex[:20]:
            xs = str(x).strip()
            if xs:
                lines.append(f"- {xs}")
    elif isinstance(ex, str) and ex.strip():
        lines.append("EXAMPLE:\n" + ex.strip())

    # 兜底：如果上面没拼出东西，就把原始 json dump 一下
    if len(lines) <= 1:
        import json
        lines = [json.dumps(obj, ensure_ascii=False)]

    return "\n".join(lines).strip()


def _build_metadata(obj: Dict[str, Any], file_posix: str, line_no: int, raw_len: int) -> Dict[str, Any]:
    """
    关键：补齐 id/source/section/title/tags —— 你 chat 输出和离线评测都靠它们。
    """
    fid = _pick_first(obj, ["id", "uid", "key"], "")
    name = _pick_first(obj, ["name", "title"], "")
    section = _pick_first(obj, ["section", "group", "category"], "")

    # 没有 id 时用可稳定的兜底（至少能唯一定位到某行）
    if not fid:
        fid = f"{file_posix}:{line_no}"

    meta: Dict[str, Any] = {
        "id": fid,
        "source": file_posix,
        "title": name,
        "section": section or name,  # 没 section 就用 name 顶上
        "tags": _normalize_tags(obj.get("tags")),
        "_file": file_posix,
        "_line": line_no,
        "_raw_len": raw_len,
    }
    return meta


def load_jsonl_as_documents(
    jsonl_path: str,
    *,
    verbose_warn_bad_json: bool = True,
) -> List[Document]:
    """
    ✅ 每一行 jsonl 作为一个 Document
    ✅ 解析 JSON，构造更适合 embedding 的 page_content
    ✅ metadata 补齐 id/source/title/section/tags（支持 chat 展示与离线评测）
    """
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path} (cwd={Path.cwd()})")

    docs: List[Document] = []
    bad_cnt = 0
    file_posix = str(p.as_posix())

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            raw = (line or "").strip()
            if not raw:
                continue

            try:
                import json
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("json line is not an object")
            except Exception:
                bad_cnt += 1
                if verbose_warn_bad_json and bad_cnt <= 20:
                    print(f"[WARN] bad json line skipped: file={jsonl_path} line={line_no}")
                continue

            meta = _build_metadata(obj, file_posix=file_posix, line_no=line_no, raw_len=len(raw))
            content = _build_page_content(obj)
            docs.append(Document(page_content=content, metadata=meta))

    if verbose_warn_bad_json and bad_cnt > 20:
        print(f"[WARN] bad json lines skipped: file={jsonl_path} count={bad_cnt} (only first 20 shown)")

    return docs


def load_many(paths: List[str]) -> List[Document]:
    """
    多个输入：.jsonl 走逐行；其他文本文件走 TextLoader 兜底
    """
    all_docs: List[Document] = []
    for p in paths:
        if p.lower().endswith(".jsonl"):
            one = load_jsonl_as_documents(p)
            print(f"[LOAD] {p} -> {len(one)} docs")
            all_docs.extend(one)
        else:
            from langchain_community.document_loaders import TextLoader

            one = TextLoader(p, encoding="utf-8").load()
            print(f"[LOAD] {p} -> {len(one)} docs")
            all_docs.extend(one)
    return all_docs
