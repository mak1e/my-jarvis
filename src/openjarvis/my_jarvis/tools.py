"""Custom OpenJarvis tools registered cleanly into ``ToolRegistry``.

These tools extend Stanford's OpenJarvis with:
- ``switch_brain`` — inspect or hot-swap JARVIS's active model/provider
- ``hardware_advisor`` — check local CPU/RAM/VRAM and fitting GGUF models
- ``hf_model_search`` — find Hugging Face GGUF models + Ollama pull commands
- ``site_security_audit`` — defensive HTTP security header & TLS posture audit
- ``repo_security_audit`` — static local codebase secret & hygiene check
- ``jarvis_memory`` — save/list/recall persistent facts in OpenJarvis's LocalFactStore
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.memory.store import TRUST_TRUSTED, LocalFactStore
from openjarvis.my_jarvis.audit import audit_local_repo, audit_website_posture
from openjarvis.my_jarvis.hardware import (
    inspect_hardware,
    recommend_models_for_hardware,
    search_huggingface_gguf,
)
from openjarvis.tools._stubs import BaseTool, ToolSpec


@ToolRegistry.register("hardware_advisor")
class HardwareAdvisorTool(BaseTool):
    """Inspect local CPU/RAM/VRAM and recommend fitting local open-weight models."""

    tool_id = "hardware_advisor"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hardware_advisor",
            description=(
                "Inspect the local machine's CPU, RAM, disk, GPU/VRAM, and Ollama status, "
                "and recommend which local GGUF / Ollama models fit this hardware."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        del params
        profile = inspect_hardware()
        recs = recommend_models_for_hardware(profile)
        payload = {
            "hardware": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
        }
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps(payload, indent=2),
            success=True,
        )


@ToolRegistry.register("hf_model_search")
class HuggingFaceModelSearchTool(BaseTool):
    """Search Hugging Face Hub for GGUF models runnable directly in Ollama."""

    tool_id = "hf_model_search"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="hf_model_search",
            description=(
                "Search Hugging Face Hub for local GGUF models (e.g. 'qwen', 'deepseek', "
                "'abliterated', 'dolphin') and return ready-to-run `ollama run hf.co/...` commands."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Model search keywords (e.g., 'qwen3', 'deepseek', 'abliterated').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-15, default 6).",
                    },
                },
                "required": ["query"],
            },
            category="models",
        )

    def execute(self, **params: Any) -> ToolResult:
        query = str(params.get("query") or "qwen").strip()
        limit = int(params.get("limit") or 6)
        data = search_huggingface_gguf(query, limit=limit)
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps(data, indent=2),
            success=True,
        )


@ToolRegistry.register("site_security_audit")
class SiteSecurityAuditTool(BaseTool):
    """Run a passive, defensive HTTP security header & TLS check on a website you own."""

    tool_id = "site_security_audit"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="site_security_audit",
            description=(
                "Perform a single passive defensive security header, cookie flag, and TLS "
                "certificate posture check on a website or SaaS URL that you own."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target HTTP/HTTPS URL you own (e.g. 'https://example.com').",
                    }
                },
                "required": ["url"],
            },
            category="security",
        )

    def execute(self, **params: Any) -> ToolResult:
        url = str(params.get("url") or "").strip()
        if not url:
            return ToolResult(
                tool_name=self.tool_id,
                content="Error: 'url' parameter is required.",
                success=False,
            )
        report = audit_website_posture(url)
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps(report.to_dict(), indent=2),
            success=not bool(report.error),
        )


@ToolRegistry.register("repo_security_audit")
class RepoSecurityAuditTool(BaseTool):
    """Scan a local project directory for leaked secrets and insecure coding patterns."""

    tool_id = "repo_security_audit"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="repo_security_audit",
            description=(
                "Run a static defensive hygiene scan on a local codebase directory to catch "
                "accidentally committed secrets and insecure coding patterns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Local filesystem path to the directory to audit.",
                    }
                },
                "required": ["path"],
            },
            category="security",
        )

    def execute(self, **params: Any) -> ToolResult:
        path = str(params.get("path") or ".").strip()
        report = audit_local_repo(path)
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps(report, indent=2),
            success="error" not in report,
        )


@ToolRegistry.register("jarvis_memory")
class JarvisMemoryTool(BaseTool):
    """Store, list, or recall persistent personal facts in OpenJarvis's LocalFactStore."""

    tool_id = "jarvis_memory"

    def __init__(self, fact_store: Optional[LocalFactStore] = None) -> None:
        self._store = fact_store or LocalFactStore()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="jarvis_memory",
            description=(
                "Manage JARVIS's persistent personal memory across sessions and brain switches. "
                "Actions: 'remember' (save a fact), 'list' (show all facts), or 'search' (filter facts)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["remember", "list", "search"],
                        "description": "Memory operation to perform.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Fact text to store (for 'remember') or query string (for 'search').",
                    },
                },
                "required": ["action"],
            },
            category="memory",
        )

    def execute(self, **params: Any) -> ToolResult:
        action = str(params.get("action") or "list").strip().lower()
        text = str(params.get("text") or "").strip()
        if action == "remember":
            if not text:
                return ToolResult(
                    tool_name=self.tool_id,
                    content="Provide non-empty text to remember.",
                    success=False,
                )
            added = self._store.add_with_trust(
                text, source="jarvis_tool", trust=TRUST_TRUSTED
            )
            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(
                    {"stored": added, "text": text, "total_facts": self._store.count()}
                ),
                success=True,
            )
        facts = [
            {
                "index": idx,
                "text": f.text,
                "source": f.source,
                "trust": f.trust or "trusted",
            }
            for idx, f in enumerate(self._store.list())
            if f.trusted_for_recall
        ]
        if action == "search" and text:
            low = text.lower()
            facts = [f for f in facts if low in f["text"].lower()]
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps({"count": len(facts), "facts": facts}, indent=2),
            success=True,
        )


@ToolRegistry.register("switch_brain")
class SwitchBrainTool(BaseTool):
    """Switch or list JARVIS's active AI brain (Ollama, Venice, Claude, DeepSeek, GPT, Gemini)."""

    tool_id = "switch_brain"

    def __init__(self, router: Any = None) -> None:
        self._router = router

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="switch_brain",
            description=(
                "List available JARVIS brains or switch the active brain mid-conversation "
                "(e.g. 'venice-deepseek', 'claude-opus', 'local-qwen', 'gemini-flash')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Brain ID or alias to switch to, or 'list' to view all brains.",
                    }
                },
                "required": ["target"],
            },
            category="system",
        )

    def execute(self, **params: Any) -> ToolResult:
        if self._router is None:
            return ToolResult(
                tool_name=self.tool_id,
                content="Brain router is not attached.",
                success=False,
            )
        target = str(params.get("target") or "list").strip()
        if target.lower() in ("list", "status", ""):
            return ToolResult(
                tool_name=self.tool_id,
                content=json.dumps(
                    {
                        "active": self._router.active_brain.to_dict(),
                        "brains": self._router.list_brains(probe_live=False),
                    },
                    indent=2,
                ),
                success=True,
            )
        result = self._router.switch_brain(target)
        return ToolResult(
            tool_name=self.tool_id,
            content=json.dumps(result, indent=2),
            success=bool(result.get("ok")),
        )


__all__ = [
    "HardwareAdvisorTool",
    "HuggingFaceModelSearchTool",
    "JarvisMemoryTool",
    "RepoSecurityAuditTool",
    "SiteSecurityAuditTool",
    "SwitchBrainTool",
]
