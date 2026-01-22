#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """读取 JSONL 文件并逐行解析。"""
    # 逐行解析 JSONL
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_tags(value: Any) -> List[str]:
    """将标签字段归一化为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def build_sections(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从方法条目构造展示区块。"""
    sections: List[Dict[str, Any]] = []
    description = (item.get("description") or "").strip()
    if description:
        sections.append({"title": "功能说明", "text": description})

    signatures = item.get("signatures") or []
    if signatures:
        sections.append({"title": "常用签名", "items": signatures[:8]})

    examples = item.get("examples") or []
    if examples:
        sections.append({"title": "示例片段", "code": examples[:5]})

    return sections


def summarize(text: str, limit: int = 60) -> str:
    """生成摘要文本。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def build_doc(item: Dict[str, Any]) -> Dict[str, Any]:
    """构造单条文档输出结构。"""
    name = (item.get("name") or item.get("id") or "").strip()
    description = (item.get("description") or "").strip()
    example_count = item.get("example_count")
    tags = normalize_tags(item.get("category"))
    subtitle_parts = []
    if example_count is not None:
        subtitle_parts.append(f"示例数 {example_count}")
    if item.get("source"):
        subtitle_parts.append(f"来源 {len(item['source'])}")
    subtitle = " · ".join(subtitle_parts) if subtitle_parts else "点击查看详情"

    return {
        "id": item.get("id") or name,
        "title": f"{name} 方法" if name else "未命名方法",
        "summary": summarize(description) if description else "暂无描述",
        "tags": tags,
        "subtitle": subtitle,
        "sections": build_sections(item),
    }


def normalize_extra_docs(raw_docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """规范化额外文档结构。"""
    out: List[Dict[str, Any]] = []
    for doc in raw_docs:
        if not isinstance(doc, dict):
            continue
        normalized = {
            "id": doc.get("id") or doc.get("title"),
            "title": doc.get("title") or "未命名文档",
            "summary": doc.get("summary") or "暂无描述",
            "tags": normalize_tags(doc.get("tags")),
            "subtitle": doc.get("subtitle") or "",
            "sections": doc.get("sections") or [],
            "order": doc.get("order", 0),
        }
        out.append(normalized)
    return out


def load_extra_docs(path: Path) -> List[Dict[str, Any]]:
    """读取并解析额外文档文件。"""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        raw_docs = payload.get("docs") or []
    elif isinstance(payload, list):
        raw_docs = payload
    else:
        raw_docs = []
    return normalize_extra_docs(raw_docs)


def build_payload(
    items: Iterable[Dict[str, Any]],
    source: str,
    extra_docs: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """构建用于前端的 docs payload。"""
    # 主列表转换为 docs
    docs = [build_doc(item) for item in items]
    # 合并额外文档
    docs.extend(extra_docs)
    # 按排序字段和标题排序
    docs.sort(key=lambda doc: (doc.get("order", 1), (doc.get("title") or "").lower()))
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source,
        "docs": docs,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build web docs json from method list JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ucc_methods_list.jsonl"),
        help="Path to method list jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/docs.json"),
        help="Output path for web docs json.",
    )
    parser.add_argument(
        "--extra-docs",
        type=Path,
        default=Path("data/ucc_official_docs.json"),
        help="Optional extra docs json to merge.",
    )
    return parser.parse_args()


def main() -> None:
    """生成前端需要的 docs.json。"""
    args = parse_args()
    # 读取额外文档
    extra_docs = load_extra_docs(args.extra_docs)
    # 构建输出 payload
    payload = build_payload(load_jsonl(args.input), str(args.input), extra_docs=extra_docs)
    # 写入输出文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
