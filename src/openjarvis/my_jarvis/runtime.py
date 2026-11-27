"""Core runtime for My-JARVIS built on top of Stanford's OpenJarvis stack.

Connects:
- ``BrainRouter`` (hot-swappable local Ollama, Venice AI, Claude, DeepSeek, GPT, Gemini)
- OpenJarvis ``OrchestratorAgent`` + ``ToolRegistry`` (built-in + custom tools)
- OpenJarvis ``LocalFactStore`` (persistent long-term memory across every brain switch)
- OpenJarvis ``EventBus`` (live telemetry & trace events)
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.events import EventBus
from openjarvis.core.paths import get_config_dir
from openjarvis.core.types import Conversation, Message, Role
from openjarvis.memory.store import TRUST_TRUSTED, LocalFactStore
from openjarvis.my_jarvis.audit import audit_local_repo, audit_website_posture
from openjarvis.my_jarvis.brains import BrainRouter, detect_brain_switch_intent
from openjarvis.my_jarvis.briefing import MorningBriefingService
from openjarvis.my_jarvis.channels_bridge import JarvisChannelManager
from openjarvis.my_jarvis.hardware import (
    inspect_hardware,
    recommend_models_for_hardware,
    search_huggingface_gguf,
)
from openjarvis.my_jarvis.mcp_hub import JarvisMCPHub
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
from openjarvis.tools._stubs import BaseTool
from openjarvis.tools.calculator import CalculatorTool, safe_eval

logger = logging.getLogger(__name__)

DEFAULT_JARVIS_PERSONA = (
    "You are J.A.R.V.I.S., a personal AI built on Stanford's OpenJarvis local-first framework. "
    "Be sharp, concise, technically accurate, and quietly witty like an aerospace copilot. "
    "Answer directly without unnecessary filler. When helpful, use your registered OpenJarvis tools "
    "(calculator, hardware_advisor, hf_model_search, site_security_audit, repo_security_audit, "
    "jarvis_memory, switch_brain, morning_briefing, mcp_hub, and any connected MCP tools). "
    "Keep spoken-friendly replies clear and structured."
)

_REMEMBER_PATTERN = re.compile(
    r"^(?:hey\s+jarvis[, ]+)?(?:please\s+)?remember(?:\s+that)?\s+(.+)$",
    re.IGNORECASE,
)


@dataclass
class TurnRecord:
    id: str
    role: str
    content: str
    timestamp: float
    brain_id: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    brain_switched_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MyJarvisRuntime:
    """Stateful conversational & agentic runtime on top of OpenJarvis."""

    def __init__(
        self,
        *,
        router: Optional[BrainRouter] = None,
        fact_store: Optional[LocalFactStore] = None,
        session_path: Optional[Path] = None,
        persona: str = DEFAULT_JARVIS_PERSONA,
        briefing_config_path: Optional[Path] = None,
        mcp_config_path: Optional[Path] = None,
    ) -> None:
        self.router = router or BrainRouter()
        self.fact_store = fact_store or LocalFactStore()
        self.bus = EventBus(record_history=True)
        self.persona = persona
        self._session_path = session_path or (
            get_config_dir() / "my_jarvis_history.json"
        )
        self.briefing_service = MorningBriefingService(
            fact_store=self.fact_store,
            config_path=briefing_config_path,
        )
        self.mcp_hub = JarvisMCPHub(config_path=mcp_config_path)
        self.speech_hub = JarvisSpeechHub()
        self.channel_manager = JarvisChannelManager(
            chat_callback=self.chat,
            bus=self.bus,
        )
        self._history: List[TurnRecord] = []
        self._load_history()

    def shutdown(self) -> None:
        """Cleanly close active MCP transports and messaging channel listeners."""
        self.channel_manager.disconnect_all()
        self.mcp_hub.close_all()

    # -- Persistence --------------------------------------------------------

    def _load_history(self) -> None:
        if not self._session_path.exists():
            return
        try:
            raw = json.loads(self._session_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._history = [
                    TurnRecord(**item)
                    for item in raw[-100:]
                    if isinstance(item, dict) and "role" in item and "content" in item
                ]
        except Exception:
            self._history = []

    def _save_history(self) -> None:
        try:
            self._session_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_path.write_text(
                json.dumps([t.to_dict() for t in self._history[-100:]], indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    def get_history(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._history]

    def clear_history(self) -> int:
        count = len(self._history)
        self._history = []
        self._save_history()
        return count

    # -- Memory management --------------------------------------------------

    def list_memories(self) -> List[Dict[str, Any]]:
        facts = self.fact_store.list()
        return [
            {
                "index": idx,
                "text": f.text,
                "source": f.source or "user",
                "created_at": f.created_at,
                "trust": f.trust or TRUST_TRUSTED,
            }
            for idx, f in enumerate(facts)
            if f.trusted_for_recall
        ]

    def add_memory(self, text: str, source: str = "user") -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            return {"ok": False, "error": "Memory text cannot be empty."}
        added = self.fact_store.add_with_trust(
            cleaned, source=source, trust=TRUST_TRUSTED
        )
        return {
            "ok": True,
            "added": added,
            "text": cleaned,
            "count": self.fact_store.count(),
        }

    def delete_memory(self, index: int) -> Dict[str, Any]:
        facts = self.fact_store.list()
        if index < 0 or index >= len(facts):
            return {"ok": False, "error": f"Invalid memory index: {index}"}
        remaining = [f for i, f in enumerate(facts) if i != index]
        self.fact_store.clear()
        for f in remaining:
            self.fact_store.add_with_trust(
                f.text,
                source=f.source,
                trust=f.trust or TRUST_TRUSTED,
            )
        return {"ok": True, "count": self.fact_store.count()}

    # -- System prompt assembly ---------------------------------------------

    def _build_system_prompt(self) -> str:
        brain = self.router.active_brain
        parts = [
            self.persona,
            f"Active brain profile: {brain.name} (model `{brain.model}` via `{brain.provider}`).",
        ]
        memories = self.list_memories()
        if memories:
            bullet_lines = "\n".join(f"- {m['text']}" for m in memories[-25:])
            parts.append(f"Persistent facts remembered about the user:\n{bullet_lines}")
        return "\n\n".join(parts)

    def _build_tools(self) -> List[BaseTool]:
        base_tools: List[BaseTool] = [
            CalculatorTool(),
            HardwareAdvisorTool(),
            HuggingFaceModelSearchTool(),
            SiteSecurityAuditTool(),
            RepoSecurityAuditTool(),
            JarvisMemoryTool(self.fact_store),
            SwitchBrainTool(self.router),
            MorningBriefingTool(self.briefing_service),
            MCPHubStatusTool(self.mcp_hub),
        ]
        # Automatically expose every tool discovered from connected MCP servers.
        base_tools.extend(self.mcp_hub.get_active_tools())
        return base_tools

    # -- Deterministic tool fast-path (works offline & gives snappy replies) -

    def _try_direct_command(
        self, text: str
    ) -> Optional[tuple[str, List[Dict[str, Any]], Optional[str]]]:
        """Handle instant slash commands, voice brain switches, and explicit memory saves."""
        stripped = text.strip()

        # 1. Voice / slash brain switch
        switch_target = detect_brain_switch_intent(stripped)
        if switch_target is not None:
            if switch_target.lower() in ("list", "options", "available"):
                brains = self.router.list_brains(probe_live=True)
                lines = ["Available JARVIS brains:"]
                for b in brains:
                    mark = "●" if b["active"] else "○"
                    state = "ready" if b["ready"] else "needs setup"
                    lines.append(
                        f"- {mark} **{b['name']}** (`{b['id']}`) — `{b['model']}` [{state}]"
                    )
                return (
                    "\n".join(lines),
                    [{"tool": "switch_brain", "action": "list"}],
                    None,
                )
            res = self.router.switch_brain(switch_target)
            if res.get("ok"):
                active = res["active_brain"]
                return (
                    res["note"],
                    [
                        {
                            "tool": "switch_brain",
                            "target": active["id"],
                            "ready": res["ready"],
                        }
                    ],
                    active["id"],
                )
            return res.get("error", "Could not switch brain."), [], None

        # 2. Explicit "remember that ..." capture
        rem_match = _REMEMBER_PATTERN.match(stripped)
        if rem_match:
            fact_text = rem_match.group(1).strip().rstrip(".")
            if fact_text:
                res = self.add_memory(fact_text, source="chat")
                reply = (
                    f'Committed to persistent OpenJarvis memory: *"{fact_text}"* '
                    f"({res['count']} total facts stored — retained across all brain switches)."
                )
                return (
                    reply,
                    [
                        {
                            "tool": "jarvis_memory",
                            "action": "remember",
                            "text": fact_text,
                        }
                    ],
                    None,
                )

        # 3. Slash & voice shortcuts
        low = stripped.lower()
        if low in (
            "/briefing",
            "morning briefing",
            "daily briefing",
            "give me my morning briefing",
            "hey jarvis give me my morning briefing",
            "jarvis morning briefing",
        ):
            rep = self.briefing_service.generate()
            return (
                rep.markdown,
                [
                    {
                        "tool": "morning_briefing",
                        "city": rep.city,
                        "events": len(rep.calendar_events),
                        "news": len(rep.news),
                    }
                ],
                None,
            )

        if low in (
            "/brains",
            "list brains",
            "what brains do you have",
            "what brain are you using",
        ):
            active = self.router.active_brain
            brains = self.router.list_brains(probe_live=True)
            ready_names = [b["name"] for b in brains if b["ready"]]
            return (
                f"Currently running on **{active.name}** (`{active.model}` via `{active.provider}`).\n\n"
                f"Ready brains right now: {', '.join(ready_names)}.\n"
                'Say *"switch brain to DeepSeek"*, *"switch to Claude"*, or *"switch to local Qwen"* anytime.',
                [{"tool": "switch_brain", "action": "status"}],
                None,
            )

        if low in ("/memory", "/memories", "what do you remember", "show my memory"):
            mems = self.list_memories()
            if not mems:
                return (
                    'Your persistent memory vault is empty. Say *"Remember that I use macOS with 32GB RAM"* to store a fact.',
                    [{"tool": "jarvis_memory", "action": "list"}],
                    None,
                )
            lines = ["Here is what I remember across all brains:"] + [
                f"- {m['text']}" for m in mems
            ]
            return "\n".join(lines), [{"tool": "jarvis_memory", "action": "list"}], None

        return None

    def _run_offline_tool_assistant(
        self, text: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Smart local tool dispatch when running on ``builtin-offline``."""
        low = text.lower().strip()

        # Hardware / VRAM check
        if any(
            k in low
            for k in (
                "hardware",
                "vram",
                "gpu",
                "ram",
                "specs",
                "what model can i run",
                "my pc",
                "my mac",
            )
        ):
            prof = inspect_hardware(self.router.ollama_host)
            recs = recommend_models_for_hardware(prof)
            optimal = [r for r in recs if r.fit == "optimal"]
            tight = [r for r in recs if r.fit == "tight"]
            gpu_desc = (
                ", ".join(f"{g.name} ({g.vram_gb:.1f} GB)" for g in prof.gpus)
                if prof.gpus
                else "CPU-only (no discrete GPU detected)"
            )
            lines = [
                f"### System Telemetry ({prof.os_name} · {prof.arch})",
                f"- **CPU**: {prof.cpu_cores_physical} physical / {prof.cpu_cores_logical} logical cores",
                f"- **RAM**: {prof.ram_total_gb:.1f} GB total ({prof.ram_available_gb:.1f} GB free)",
                f"- **GPU / VRAM**: {gpu_desc}",
                f"- **Ollama**: {'Running (' + str(len(prof.ollama_models)) + ' models)' if prof.ollama_running else 'Not running'}",
                "",
                "**Best-fit local models for this hardware:**",
            ]
            for r in (optimal or tight)[:4]:
                lines.append(
                    f"- **{r.name}** ({r.fit.upper()}) — `{r.ollama_cmd}` — {r.fit_reason}"
                )
            if not optimal:
                lines.append(
                    "\nFor heavier reasoning (DeepSeek / 27B–70B+), switch to **Venice DeepSeek** or a Cloud brain in the **Brains** tab."
                )
            return "\n".join(lines), [
                {
                    "tool": "hardware_advisor",
                    "ram_gb": prof.ram_total_gb,
                    "vram_gb": prof.total_vram_gb,
                }
            ]

        # Hugging Face search
        hf_match = re.search(
            r"(?:search\s+(?:hugging\s*face|hf)\s+(?:for\s+)?|find\s+gguf\s+)(.+)$",
            text,
            re.IGNORECASE,
        )
        if hf_match or "huggingface" in low or "gguf" in low:
            q = (
                hf_match.group(1).strip()
                if hf_match
                else (
                    "abliterated"
                    if "uncensored" in low or "abliterated" in low
                    else "qwen"
                )
            )
            res = search_huggingface_gguf(q, limit=5)
            lines = [f"### Hugging Face GGUF Matches for `{q}` ({res['source']}):"]
            for m in res["models"]:
                lines.append(f"- **{m['repo_id']}** → `{m['ollama_run_cmd']}`")
            return "\n".join(lines), [
                {"tool": "hf_model_search", "query": q, "count": len(res["models"])}
            ]

        # Defensive site security audit
        url_match = re.search(
            r"(https?://[^\s]+|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?)",
            text,
            re.IGNORECASE,
        )
        if (
            "audit" in low
            or "security" in low
            or "headers" in low
            or "check site" in low
        ) and url_match:
            target_url = url_match.group(1).rstrip(".,!?)")
            rep = audit_website_posture(target_url)
            if rep.error:
                return f"Defensive audit could not reach `{target_url}`: {rep.error}", [
                    {
                        "tool": "site_security_audit",
                        "url": target_url,
                        "error": rep.error,
                    }
                ]
            lines = [
                f"### Defensive Security Posture: `{rep.final_url}`",
                f"- **Grade**: **{rep.grade}** ({rep.score}/100) · HTTP {rep.status_code}",
                f"- **TLS**: {rep.tls.get('protocol', 'none')} ({rep.tls.get('days_remaining', '?')} days remaining)"
                if rep.tls.get("enabled")
                else "- **TLS**: Not HTTPS",
            ]
            if rep.findings:
                lines.append("\n**Top Remediation Items:**")
                for f in rep.findings[:6]:
                    lines.append(
                        f"- `[{f.severity.upper()}]` **{f.title}** — `{f.remediation}`"
                    )
            else:
                lines.append("\nAll core HTTP security headers are present!")
            return "\n".join(lines), [
                {
                    "tool": "site_security_audit",
                    "url": rep.final_url,
                    "grade": rep.grade,
                    "score": rep.score,
                }
            ]

        # Local repo security audit
        if "audit repo" in low or "scan repo" in low or "scan code" in low:
            rep = audit_local_repo(".")
            lines = [
                f"### Local Codebase Hygiene Scan: `{rep.get('path')}`",
                f"- **Files scanned**: {rep.get('files_scanned', 0)} · **Risk**: **{rep.get('risk_level', 'LOW')}**",
                f"- **Findings**: {rep.get('summary', {}).get('total', 0)} total "
                f"({rep.get('summary', {}).get('high', 0)} high, {rep.get('summary', {}).get('medium', 0)} medium)",
            ]
            for f in rep.get("findings", [])[:6]:
                lines.append(
                    f"- `[{f['severity'].upper()}]` **{f['title']}** at `{f['location']}`"
                )
            return "\n".join(lines), [
                {
                    "tool": "repo_security_audit",
                    "path": rep.get("path"),
                    "risk": rep.get("risk_level"),
                }
            ]

        # Calculator fast-path
        calc_match = re.match(
            r"^(?:calc(?:ulate)?|compute|what(?:'s| is))\s+([0-9\s\+\-\*/\^\(\)\.a-zA-Z,]+)\??$",
            stripped := text.strip(),
            re.IGNORECASE,
        )
        if calc_match:
            expr = calc_match.group(1).strip()
            if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/^("):
                try:
                    val = safe_eval(expr)
                    return f"`{expr}` = **{val:g}**", [
                        {"tool": "calculator", "expression": expr, "result": val}
                    ]
                except Exception:
                    pass

        # Default offline engine response
        eng = self.router.get_engine_for(self.router.active_brain)
        res = eng.generate([Message(role=Role.USER, content=stripped)])
        return res["content"], []

    # -- Main Turn Execution ------------------------------------------------

    def chat(
        self,
        user_text: str,
        *,
        brain_id: Optional[str] = None,
        use_tools: bool = True,
    ) -> Dict[str, Any]:
        """Process one user turn with live brain hot-swapping, memory, and tools."""
        t0 = time.perf_counter()
        if brain_id and brain_id != self.router.active_brain.id:
            self.router.switch_brain(brain_id)

        user_turn = TurnRecord(
            id=str(uuid.uuid4())[:8],
            role="user",
            content=user_text,
            timestamp=time.time(),
            brain_id=self.router.active_brain.id,
        )
        self._history.append(user_turn)

        # 1. Check direct voice/slash commands (brain switching, memory save/list)
        direct = self._try_direct_command(user_text)
        if direct is not None:
            reply_text, tools_used, switched_to = direct
            brain = self.router.active_brain
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            assistant_turn = TurnRecord(
                id=str(uuid.uuid4())[:8],
                role="assistant",
                content=reply_text,
                timestamp=time.time(),
                brain_id=brain.id,
                model=brain.model,
                provider=brain.provider,
                latency_ms=elapsed_ms,
                tools_used=tools_used,
                brain_switched_to=switched_to,
            )
            self._history.append(assistant_turn)
            self._save_history()
            return {
                "reply": assistant_turn.to_dict(),
                "active_brain": brain.to_dict(),
                "brain_switched": switched_to is not None,
            }

        brain = self.router.active_brain
        tools_used: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        reply_text = ""

        # 2. If brain is offline core OR unconfigured external brain, degrade gracefully (zero 500s!)
        if brain.provider == "jarvis_offline" or not self.router.is_brain_configured(
            brain
        ):
            if brain.provider != "jarvis_offline":
                setup_hint = (
                    f"Start Ollama (`ollama serve && ollama pull {brain.model}`)"
                    if brain.provider == "ollama"
                    else f"Add `{brain.env_var}` in the **Brains** tab"
                )
                fallback_body, tools_used = self._run_offline_tool_assistant(user_text)
                if tools_used:
                    reply_text = (
                        f"*(Executed via local OpenJarvis tool — `{brain.name}` needs setup: {setup_hint})*\n\n"
                        + fallback_body
                    )
                else:
                    reply_text = (
                        f"**{brain.name}** (`{brain.model}`) is selected, but not connected yet.\n\n"
                        f"**Quick Fix:** {setup_hint}.\n\n"
                        "Or switch to another ready brain (say `switch brain to offline` to run local tools right now)."
                    )
            else:
                reply_text, tools_used = self._run_offline_tool_assistant(user_text)
        else:
            # 3. Live inference via Stanford OpenJarvis Engine + OrchestratorAgent!
            engine = self.router.get_engine_for(brain)
            system_prompt = self._build_system_prompt()
            prior_turns: List[Message] = []
            for turn in self._history[-17:-1]:
                role = Role.USER if turn.role == "user" else Role.ASSISTANT
                prior_turns.append(Message(role=role, content=turn.content))

            try:
                if use_tools and brain.supports_tools:
                    tools = self._build_tools()
                    agent = OrchestratorAgent(
                        engine=engine,
                        model=brain.model,
                        tools=tools,
                        bus=self.bus,
                        max_turns=6,
                        system_prompt=system_prompt,
                    )
                    ctx = AgentContext(conversation=Conversation(messages=prior_turns))
                    result = agent.run(user_text, context=ctx)
                    reply_text = result.content or ""
                    for tr in result.tool_results or []:
                        tools_used.append(
                            {
                                "tool": tr.tool_name,
                                "success": tr.success,
                                "preview": (tr.content or "")[:240],
                            }
                        )
                    usage = (result.metadata or {}).get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                else:
                    msgs = (
                        [Message(role=Role.SYSTEM, content=system_prompt)]
                        + prior_turns
                        + [Message(role=Role.USER, content=user_text)]
                    )
                    raw = engine.generate(msgs, model=brain.model)
                    reply_text = str(raw.get("content") or "")
                    usage = raw.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            except Exception as exc:
                logger.warning(
                    "Engine %s (%s) failed: %s", brain.provider, brain.model, exc
                )
                # Graceful fallback — never crash the UI with a raw 500!
                reply_text = (
                    f"⚠️ **{brain.name}** (`{brain.model}`) returned an error:\n"
                    f"`{exc}`\n\n"
                    "**Suggested Fix:** Verify your API key / model ID in the **Brains** tab, "
                    "or make sure Ollama has the model pulled (`ollama pull "
                    f"{brain.model}`)."
                )

        # Refresh active brain in case the agent called `switch_brain` during its tool loop!
        new_brain = self.router.active_brain
        switched_to = new_brain.id if new_brain.id != brain.id else None

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        est_cost = round(
            (prompt_tokens / 1_000_000.0) * brain.cost_per_1m_in
            + (completion_tokens / 1_000_000.0) * brain.cost_per_1m_out,
            6,
        )

        assistant_turn = TurnRecord(
            id=str(uuid.uuid4())[:8],
            role="assistant",
            content=reply_text,
            timestamp=time.time(),
            brain_id=new_brain.id,
            model=new_brain.model,
            provider=new_brain.provider,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=est_cost,
            tools_used=tools_used,
            brain_switched_to=switched_to,
        )
        self._history.append(assistant_turn)
        self._save_history()

        return {
            "reply": assistant_turn.to_dict(),
            "active_brain": new_brain.to_dict(),
            "brain_switched": switched_to is not None,
        }

    # -- Diagnostic Doctor (inspired by Video 2 step-by-step setup & fixes) -

    def run_doctor(self) -> Dict[str, Any]:
        """Run comprehensive OpenJarvis + My-JARVIS health checks with actionable fixes."""
        import sys

        import openjarvis

        hw = inspect_hardware(self.router.ollama_host)
        providers = self.router.get_provider_status()
        configured_cloud = [
            p["provider"] for p in providers if p["type"] != "local" and p["configured"]
        ]

        checks = [
            {
                "name": "Python Runtime",
                "status": "ok" if (3, 11) <= sys.version_info[:2] < (3, 14) else "warn",
                "detail": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "fix": "Use Python 3.11–3.13 for prebuilt OpenJarvis wheels."
                if not ((3, 11) <= sys.version_info[:2] < (3, 14))
                else "",
            },
            {
                "name": "Stanford OpenJarvis Core",
                "status": "ok",
                "detail": f"openjarvis {openjarvis.__version__} installed",
                "fix": "",
            },
            {
                "name": "Local Ollama Engine",
                "status": "ok" if hw.ollama_running else "warn",
                "detail": (
                    f"Online at {self.router.ollama_host} ({len(hw.ollama_models)} models pulled)"
                    if hw.ollama_running
                    else f"Not reachable at {self.router.ollama_host}"
                ),
                "fix": ""
                if hw.ollama_running
                else "Install Ollama (https://ollama.com) then run: ollama serve && ollama pull qwen3:8b",
            },
            {
                "name": "Venice / Cloud Brains",
                "status": "ok" if configured_cloud else "warn",
                "detail": (
                    f"Configured: {', '.join(configured_cloud)}"
                    if configured_cloud
                    else "No cloud/Venice API keys set yet"
                ),
                "fix": ""
                if configured_cloud
                else "Optional: Add VENICE_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in the Brains tab.",
            },
            {
                "name": "Persistent Memory Store",
                "status": "ok",
                "detail": f"LocalFactStore active ({self.fact_store.count()} trusted facts)",
                "fix": "",
            },
            {
                "name": "Hardware Headroom",
                "status": "ok",
                "detail": (
                    f"{hw.ram_total_gb:.1f} GB RAM · "
                    f"{f'{hw.total_vram_gb:.1f} GB VRAM ({hw.accelerator})' if hw.total_vram_gb > 0 else 'CPU mode'}"
                ),
                "fix": "",
            },
            {
                "name": "MCP Server Hub",
                "status": "ok",
                "detail": (
                    f"{len(self.mcp_hub.list_servers()['servers'])} servers configured · "
                    f"{self.mcp_hub.list_servers()['total_mcp_tools']} active tools"
                ),
                "fix": "",
            },
            {
                "name": "Messaging Channels",
                "status": "ok",
                "detail": (
                    f"{sum(1 for c in self.channel_manager.list_channels() if c['status'] == 'connected')} "
                    "connected (Webhook always active)"
                ),
                "fix": "",
            },
            {
                "name": "Neural Voice Engines",
                "status": "ok"
                if self.speech_hub.status()["neural_tts_ready"]
                else "warn",
                "detail": (
                    "Neural TTS ready"
                    if self.speech_hub.status()["neural_tts_ready"]
                    else "Using Browser Web Speech API (optional neural TTS not configured yet)"
                ),
                "fix": ""
                if self.speech_hub.status()["neural_tts_ready"]
                else "Optional: `uv sync --extra voice` (Kokoro local TTS) or add OPENAI_API_KEY / ELEVENLABS_API_KEY.",
            },
        ]
        return {
            "ok": all(c["status"] != "fail" for c in checks),
            "active_brain": self.router.active_brain.to_dict(),
            "checks": checks,
            "hardware": hw.to_dict(),
        }


__all__ = ["DEFAULT_JARVIS_PERSONA", "MyJarvisRuntime", "TurnRecord"]
