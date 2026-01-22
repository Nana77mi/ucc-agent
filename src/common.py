# src/common.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件，失败时返回空字典。"""
    # 解析配置路径
    p = Path(path)
    # 文件不存在时返回默认值
    if not p.exists():
        return {}
    # 读取并解析 YAML 内容
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f) or {}


def read_text(path: str, default: str = "") -> str:
    """读取文本文件内容，不存在时返回默认值。"""
    # 解析目标路径
    p = Path(path)
    # 文件不存在时返回默认值
    if not p.exists():
        return default
    # 读取文本并去除首尾空白
    return p.read_text(encoding="utf-8", errors="ignore").strip()
