from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def _get_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("models", {}) or {}


def _get_provider(cfg: Dict[str, Any], key: str, default: str) -> str:
    return str(_get_cfg(cfg).get(key, default)).strip().lower()


def _get_base_url(cfg: Dict[str, Any]) -> str | None:
    models_cfg = _get_cfg(cfg)
    return models_cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")


def _get_api_key(cfg: Dict[str, Any]) -> str | None:
    models_cfg = _get_cfg(cfg)
    return models_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")


def build_embeddings(cfg: Dict[str, Any]) -> Embeddings:
    provider = _get_provider(cfg, "embed_provider", "ollama")
    model = str(_get_cfg(cfg).get("embed_model", "nomic-embed-text:latest"))

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model)

    if provider in {"openai", "qwen"}:
        from langchain_openai import OpenAIEmbeddings

        base_url = _get_base_url(cfg)
        api_key = _get_api_key(cfg)
        return OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unsupported embed_provider: {provider}")


def build_llm(cfg: Dict[str, Any], *, temperature: float) -> BaseChatModel:
    provider = _get_provider(cfg, "llm_provider", "ollama")
    model = str(_get_cfg(cfg).get("llm_model", "qwen3:4b"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature)

    if provider in {"openai", "qwen"}:
        from langchain_openai import ChatOpenAI

        base_url = _get_base_url(cfg)
        api_key = _get_api_key(cfg)
        return ChatOpenAI(model=model, temperature=temperature, base_url=base_url, api_key=api_key)

    raise ValueError(f"Unsupported llm_provider: {provider}")
