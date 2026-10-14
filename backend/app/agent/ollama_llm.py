from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


def create_ollama_llm(
    model: str = "llama3.2",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.4,
    num_ctx: int = 8192,
    num_predict: int = 1024,
    **kwargs: Any,
) -> ChatOllama:
    """Create a ChatOllama LLM instance configured for JARVIS.

    Args:
        model: Ollama model name (e.g., 'llama3.2', 'mistral', 'gemma2').
        base_url: Ollama server URL (must be local loopback).
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        num_ctx: Context window size in tokens.
        num_predict: Maximum tokens to generate per response.

    Returns:
        A configured ChatOllama instance ready for LangGraph.
    """
    logger.info("Initializing Ollama LLM: model=%s, url=%s", model, base_url)

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        # Keep-alive prevents model unloading between turns
        keep_alive="10m",
        **kwargs,
    )
