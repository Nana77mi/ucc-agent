#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def build_sections(item: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def build_doc(item: Dict[str, Any]) -> Dict[str, Any]:
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
    docs = [build_doc(item) for item in items]
    docs.extend(extra_docs)
    docs.sort(key=lambda doc: (doc.get("order", 1), (doc.get("title") or "").lower()))
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source,
        "docs": docs,
    }


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    extra_docs = load_extra_docs(args.extra_docs)
    payload = build_payload(load_jsonl(args.input), str(args.input), extra_docs=extra_docs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
