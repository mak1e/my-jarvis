"""My-JARVIS — Custom multi-brain personal AI built on top of Stanford OpenJarvis."""

from __future__ import annotations

from openjarvis.core.registry import EngineRegistry, ToolRegistry
from openjarvis.my_jarvis.brains import (
    BrainProfile,
    BrainRouter,
    BuiltinOfflineEngine,
    detect_brain_switch_intent,
)
from openjarvis.my_jarvis.briefing import MorningBriefingService
from openjarvis.my_jarvis.channels_bridge import JarvisChannelManager
from openjarvis.my_jarvis.mcp_hub import JarvisMCPHub
from openjarvis.my_jarvis.runtime import MyJarvisRuntime
from openjarvis.my_jarvis.speech_hub import JarvisSpeechHub
from openjarvis.my_jarvis.tools import (
    HardwareAdvisorTool,
    HuggingFaceModelSearchTool,
    JarvisMemoryTool,
    MCPHubStatusTool,
    MorningBriefingTool,
    RepoSecurityAuditTool,
    SiteSecurityAuditTool,
    SwitchBrainTool,
)
from openjarvis.my_jarvis.venice_engine import VeniceEngine


def register_my_jarvis_primitives() -> None:
    """Idempotently register My-JARVIS engines and tools into OpenJarvis."""
    if not EngineRegistry.contains("venice"):
        EngineRegistry.register("venice")(VeniceEngine)
    if not EngineRegistry.contains("jarvis_offline"):
        EngineRegistry.register("jarvis_offline")(BuiltinOfflineEngine)
    for key, cls in (
        ("hardware_advisor", HardwareAdvisorTool),
        ("hf_model_search", HuggingFaceModelSearchTool),
        ("site_security_audit", SiteSecurityAuditTool),
        ("repo_security_audit", RepoSecurityAuditTool),
        ("jarvis_memory", JarvisMemoryTool),
        ("switch_brain", SwitchBrainTool),
        ("morning_briefing", MorningBriefingTool),
        ("mcp_hub", MCPHubStatusTool),
    ):
        if not ToolRegistry.contains(key):
            ToolRegistry.register(key)(cls)


register_my_jarvis_primitives()

__all__ = [
    "BrainProfile",
    "BrainRouter",
    "BuiltinOfflineEngine",
    "JarvisChannelManager",
    "JarvisMCPHub",
    "JarvisSpeechHub",
    "MorningBriefingService",
    "MyJarvisRuntime",
    "VeniceEngine",
    "detect_brain_switch_intent",
    "register_my_jarvis_primitives",
]
