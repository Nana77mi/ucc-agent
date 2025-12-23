"""Data loading utilities for JSONL datasets."""

from pathlib import Path
from typing import Iterable, Dict
import json


def load_jsonl(path: Path) -> Iterable[Dict]:
    """Load records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        An iterable of JSON objects.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
