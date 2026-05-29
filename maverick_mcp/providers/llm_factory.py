"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances with intelligent model selection.
"""

import logging
import os
from typing import Any

from langchain_community.llms import FakeListLLM

from maverick_mcp.providers.openrouter_provider import (
    TaskType,
    get_openrouter_llm,
)

logger = logging.getLogger(__name__)


def get_provider_llm() -> Any:
    """Thin LLM provider abstraction that supports any OpenAI-compatible API via .env:
    - LLM_PROVIDER=openai | deepseek | anthropic | openrouter
    - LLM_API_KEY=sk-...
    - LLM_MODEL=gpt-4o-mini | deepseek-chat | claude-3-sonnet | etc.
    - LLM_BASE_URL=optional (for DeepSeek, OpenRouter, or local endpoints)
    """
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")

    if not provider:
        return None

    logger.info(f"Initializing configured LLM Provider: {provider} (model: {model})")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
            openai_api_base=base_url,
            temperature=0.3,
            streaming=False,
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "deepseek-chat",
            openai_api_key=api_key or os.getenv("DEEPSEEK_API_KEY") or api_key,
            openai_api_base=base_url or "https://api.deepseek.com",
            temperature=0.3,
            streaming=False,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-3-haiku-20240307",
            anthropic_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "google/gemini-2.5-flash",
            openai_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=base_url or "https://openrouter.ai/api/v1",
            temperature=0.3,
            streaming=False,
        )

    logger.warning(
        f"Unknown LLM_PROVIDER '{provider}'. Falling back to original routing."
    )
    return None


def get_llm(
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,  # Default to cost-effective
    prefer_quality: bool = False,
    model_override: str | None = None,
) -> Any:
    """Create and return an LLM instance with intelligent model selection.

    Args:
        task_type: Type of task to optimize model selection for
        prefer_fast: Prioritize speed over quality
        prefer_cheap: Prioritize cost over quality (default True)
        prefer_quality: Use premium models regardless of cost
        model_override: Override automatic model selection

    Returns:
        An LLM instance optimized for the task.
    """
    # 1. Check for LLM_PROVIDER from .env first (thin provider agnostic abstraction)
    provider_llm = get_provider_llm()
    if provider_llm is not None:
        return provider_llm

    # 2. Check for OpenRouter (preferred)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        logger.info(
            f"Using OpenRouter with intelligent model selection for task: {task_type}"
        )
        return get_openrouter_llm(
            api_key=openrouter_api_key,
            task_type=task_type,
            prefer_fast=prefer_fast,
            prefer_cheap=prefer_cheap,
            prefer_quality=prefer_quality,
            model_override=model_override,
        )

    # 3. Fallback to OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        logger.info("Falling back to OpenAI API")
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=False)
        except ImportError:
            pass

    # 4. Fallback to Anthropic
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        logger.info("Falling back to Anthropic API")
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
        except ImportError:
            pass

    # 5. Final fallback to fake LLM for testing
    logger.warning("No LLM API keys found - using FakeListLLM for testing")
    return FakeListLLM(
        responses=[
            "Mock analysis response for testing purposes.",
            "This is a simulated LLM response.",
            "Market analysis: Moderate bullish sentiment detected.",
        ]
    )
