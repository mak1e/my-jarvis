"""Venice AI OpenAI-compatible inference engine for OpenJarvis.

Connects OpenJarvis to Venice AI (https://api.venice.ai/api/v1) for private,
uncensored, and open-weight cloud inference (e.g. DeepSeek, Qwen, Llama,
Venice Uncensored) while inheriting OpenJarvis's full streaming, tool-calling,
and telemetry pipeline.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Dict, List

import httpx

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._openai_compat import _OpenAICompatibleEngine

logger = logging.getLogger(__name__)

VENICE_DEFAULT_MODELS: List[str] = [
    "deepseek-r1-671b",
    "venice-uncensored",
    "qwen3-235b",
    "llama-3.3-70b",
    "mistral-31-24b",
    "qwen-2.5-coder-32b",
]


def _read_venice_key() -> str | None:
    """Resolve ``VENICE_API_KEY`` from process env or ``cloud-keys.env``."""
    env_key = os.environ.get("VENICE_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file = get_config_dir() / "cloud-keys.env"
    if key_file.exists():
        try:
            for raw in key_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "VENICE_API_KEY":
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
        except OSError:
            pass
    return None


@EngineRegistry.register("venice")
class VeniceEngine(_OpenAICompatibleEngine):
    """OpenJarvis inference engine backed by the Venice AI OpenAI-compatible API."""

    engine_id = "venice"
    is_cloud = True
    _default_host = "https://api.venice.ai"
    _api_prefix = "/api/v1"

    def __init__(
        self,
        host: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        disable_venice_system_prompt: bool = True,
    ) -> None:
        resolved_key = api_key or _read_venice_key()
        super().__init__(host=host, api_key=resolved_key, timeout=timeout)
        self._disable_venice_system_prompt = disable_venice_system_prompt

    def _resolve_model_id(self, model: str) -> str:
        # Allow users to prefix with "venice/" for explicit disambiguation.
        if model.startswith("venice/"):
            return model[len("venice/") :]
        return model

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self._disable_venice_system_prompt and "venice_parameters" not in kwargs:
            kwargs["venice_parameters"] = {"include_venice_system_prompt": False}
        return super().generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def can_serve(self, model: str) -> bool:
        return bool(self._api_key)

    def health(self) -> bool:
        """Return True when ``VENICE_API_KEY`` is configured and API responds."""
        if not self._api_key:
            return False
        try:
            resp = self._client.get(f"{self._api_prefix}/models", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> List[str]:
        """List available Venice AI models, falling back to known catalog IDs."""
        if not self._api_key:
            return list(VENICE_DEFAULT_MODELS)
        try:
            resp = self._client.get(f"{self._api_prefix}/models", timeout=8.0)
            if resp.status_code == 200:
                payload = resp.json()
                ids = [
                    str(item["id"])
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                ]
                if ids:
                    return ids
        except Exception as exc:
            logger.debug("Venice model discovery failed: %s", exc)
        return list(VENICE_DEFAULT_MODELS)


__all__ = ["VENICE_DEFAULT_MODELS", "VeniceEngine"]
