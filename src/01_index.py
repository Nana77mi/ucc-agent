"""Index building entry point."""

from pathlib import Path
from loader_jsonl import load_jsonl


def build_index(data_path: Path, output_dir: Path) -> None:
    """Placeholder function for building an index."""
    records = list(load_jsonl(data_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.info").write_text(f"Indexed {len(records)} records", encoding="utf-8")


if __name__ == "__main__":
    build_index(Path("../data/ucc_rag.jsonl"), Path("../index/faiss_ucc"))
