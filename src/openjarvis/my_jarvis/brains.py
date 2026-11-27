"""Hot-swappable Multi-Brain Router built on top of OpenJarvis engines.

Supports switching JARVIS's active model/provider mid-conversation via voice
command, chat command (``/brain <id>``), OpenJarvis tool call, or web HUD
selector across:
- Local Ollama models (Qwen, DeepSeek-R1, Dolphin, or any pulled GGUF model)
- Venice AI API (DeepSeek / Venice Uncensored / open-weight models)
- Cloud frontier providers via OpenJarvis CloudEngine (Anthropic Claude,
  OpenAI GPT, Google Gemini, DeepSeek API, OpenRouter)
- Built-in Offline Tool Assistant (always available when no GPU/keys exist yet)
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import EngineRegistry
from openjarvis.core.types import Message
from openjarvis.engine._base import InferenceEngine
from openjarvis.my_jarvis.venice_engine import VeniceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in offline fallback engine (ensures zero-crash first run)
# ---------------------------------------------------------------------------


@EngineRegistry.register("jarvis_offline")
class BuiltinOfflineEngine(InferenceEngine):
    """Lightweight deterministic engine when no Ollama/cloud backend is connected yet."""

    engine_id = "jarvis_offline"
    is_cloud = False

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str = "jarvis-offline-core",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        last_user = ""
        for msg in reversed(messages):
            if msg.role.value == "user" and msg.content:
                last_user = msg.content.strip()
                break
        reply = (
            "Standing by in **Local Command Mode** (no live Ollama server or cloud API key "
            "is active yet). I can still run your local OpenJarvis tools directly — try:\n"
            "- `check my hardware` — inspect CPU/RAM/VRAM & fitting local GGUF models\n"
            "- `search huggingface for qwen` — find GGUF models with Ollama pull commands\n"
            "- `audit https://example.com` — defensive HTTP security header & TLS check\n"
            "- `remember that I prefer concise replies` — save to persistent OpenJarvis memory\n"
            "- `calculate (137 * 42) + sqrt(144)` — run OpenJarvis's safe AST calculator\n"
            "- `switch brain to venice` / `claude` / `local qwen` — hot-swap brains once configured\n\n"
            f'To unlock full conversational AI for *"{last_user[:80]}"*, open the **Brains** tab '
            "to connect Ollama (free & local) or add a Venice / Claude / OpenAI / Gemini key."
        )
        prompt_est = max(1, sum(len(m.text) for m in messages) // 4)
        comp_est = max(1, len(reply) // 4)
        return {
            "content": reply,
            "usage": {
                "prompt_tokens": prompt_est,
                "completion_tokens": comp_est,
                "total_tokens": prompt_est + comp_est,
            },
            "model": model,
            "finish_reason": "stop",
        }

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str = "jarvis-offline-core",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        res = self.generate(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        yield res["content"]

    def list_models(self) -> List[str]:
        return ["jarvis-offline-core"]

    def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Brain Profiles
# ---------------------------------------------------------------------------


@dataclass
class BrainProfile:
    id: str
    name: str
    provider: str  # "ollama" | "venice" | "cloud" | "llamacpp" | "lmstudio" | "jarvis_offline"
    model: str
    description: str
    tier: str  # "local" | "uncensored" | "frontier" | "fallback"
    aliases: List[str] = field(default_factory=list)
    env_var: str = ""  # Required API key env var for cloud/venice
    host: str = ""  # Optional endpoint URL override (e.g. Ollama/vLLM host)
    cost_per_1m_in: float = 0.0
    cost_per_1m_out: float = 0.0
    supports_tools: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_BRAINS: List[BrainProfile] = [
    BrainProfile(
        id="local-qwen",
        name="Local Qwen 3 (8B)",
        provider="ollama",
        model="qwen3:8b",
        description="Free, 100% private local brain via Ollama — Stanford OpenJarvis default.",
        tier="local",
        aliases=["local", "ollama", "qwen", "qwen3", "local qwen", "free"],
        host="http://localhost:11434",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
    ),
    BrainProfile(
        id="local-qwen-27b",
        name="Local Qwen 3 (32B Heavy)",
        provider="ollama",
        model="qwen3:32b",
        description="High-intelligence local workstation brain (24 GB+ VRAM / 32 GB Mac).",
        tier="local",
        aliases=["qwen 27b", "qwen 32b", "heavy local", "big qwen"],
        host="http://localhost:11434",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
    ),
    BrainProfile(
        id="local-deepseek",
        name="Local DeepSeek-R1 (8B)",
        provider="ollama",
        model="deepseek-r1:8b",
        description="Local reasoning model running on your hardware via Ollama.",
        tier="local",
        aliases=["local deepseek", "deepseek local", "r1 local"],
        host="http://localhost:11434",
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
    ),
    BrainProfile(
        id="venice-deepseek",
        name="Venice DeepSeek (Uncensored API)",
        provider="venice",
        model="deepseek-r1-671b",
        description="Private, uncensored DeepSeek via Venice AI OpenAI-compatible API.",
        tier="uncensored",
        aliases=[
            "venice",
            "venice deepseek",
            "deepseek",
            "deepseek flash",
            "deepseek v4",
            "uncensored",
            "uncensored deepseek",
        ],
        env_var="VENICE_API_KEY",
        host="https://api.venice.ai",
        cost_per_1m_in=0.50,
        cost_per_1m_out=2.00,
    ),
    BrainProfile(
        id="venice-uncensored",
        name="Venice Uncensored (Open-Weight)",
        provider="venice",
        model="venice-uncensored",
        description="Direct unaligned Venice open-weight model with zero Venice system prompt injection.",
        tier="uncensored",
        aliases=["venice uncensored", "abliterated", "no filter"],
        env_var="VENICE_API_KEY",
        host="https://api.venice.ai",
        cost_per_1m_in=0.25,
        cost_per_1m_out=1.00,
    ),
    BrainProfile(
        id="claude-opus",
        name="Claude Opus 4.6",
        provider="cloud",
        model="claude-opus-4-6",
        description="Anthropic flagship reasoning & autonomous coding brain.",
        tier="frontier",
        aliases=["opus", "claude opus", "anthropic opus"],
        env_var="ANTHROPIC_API_KEY",
        cost_per_1m_in=5.00,
        cost_per_1m_out=25.00,
    ),
    BrainProfile(
        id="claude-sonnet",
        name="Claude Sonnet 4.6",
        provider="cloud",
        model="claude-sonnet-4-6",
        description="Fast, crisp everyday frontier brain from Anthropic.",
        tier="frontier",
        aliases=["claude", "sonnet", "claude sonnet", "anthropic"],
        env_var="ANTHROPIC_API_KEY",
        cost_per_1m_in=3.00,
        cost_per_1m_out=15.00,
    ),
    BrainProfile(
        id="deepseek-cloud",
        name="DeepSeek V4 Flash (Direct API)",
        provider="cloud",
        model="deepseek-v4-flash",
        description="Direct DeepSeek API via OpenJarvis native cloud engine.",
        tier="frontier",
        aliases=["deepseek api", "deepseek cloud", "v4 flash"],
        env_var="DEEPSEEK_API_KEY",
        cost_per_1m_in=0.27,
        cost_per_1m_out=1.10,
    ),
    BrainProfile(
        id="gpt-5",
        name="OpenAI GPT-5 Mini",
        provider="cloud",
        model="gpt-5-mini",
        description="Fast OpenAI multimodal & tool-calling model.",
        tier="frontier",
        aliases=["gpt", "openai", "gpt5", "gpt-5", "chatgpt"],
        env_var="OPENAI_API_KEY",
        cost_per_1m_in=0.25,
        cost_per_1m_out=2.00,
    ),
    BrainProfile(
        id="gemini-flash",
        name="Google Gemini 2.5 Flash",
        provider="cloud",
        model="gemini-2.5-flash",
        description="High-speed Google Gemini cloud brain.",
        tier="frontier",
        aliases=["gemini", "google", "gemini flash"],
        env_var="GEMINI_API_KEY",
        cost_per_1m_in=0.30,
        cost_per_1m_out=2.50,
    ),
    BrainProfile(
        id="openrouter-auto",
        name="OpenRouter Auto",
        provider="cloud",
        model="openrouter/auto",
        description="Access 200+ open & frontier models through one OpenRouter key.",
        tier="frontier",
        aliases=["openrouter", "router"],
        env_var="OPENROUTER_API_KEY",
        cost_per_1m_in=0.50,
        cost_per_1m_out=1.50,
    ),
    BrainProfile(
        id="builtin-offline",
        name="JARVIS Core (Offline Tools)",
        provider="jarvis_offline",
        model="jarvis-offline-core",
        description="Always-on built-in command core for local tools, memory, hardware & audits without external APIs.",
        tier="fallback",
        aliases=["offline", "fallback", "demo", "core", "built-in", "builtin"],
        cost_per_1m_in=0.0,
        cost_per_1m_out=0.0,
    ),
]


PROVIDER_ENV_KEYS: Dict[str, str] = {
    "venice": "VENICE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


# ---------------------------------------------------------------------------
# Voice / Natural-Language Brain Switch Detection
# ---------------------------------------------------------------------------

_SWITCH_PATTERNS = [
    re.compile(r"^/brain\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"(?:hey\s+jarvis[, ]+)?(?:please\s+)?(?:switch|change|swap|set|replace)\s+"
        r"(?:your\s+|jarvis(?:'s)?\s+)?(?:brain|model|engine|mind)\s+(?:to|with|over\s+to)\s+(.+?)[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:hey\s+jarvis[, ]+)?(?:please\s+)?(?:switch|swap)\s+to\s+(.+?)(?:\s+brain|\s+model)?[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:hey\s+jarvis[, ]+)?use\s+(?:the\s+)?(.+?)\s+(?:brain|model)[.!?]*$",
        re.IGNORECASE,
    ),
]


def detect_brain_switch_intent(text: str) -> Optional[str]:
    """Return the requested brain target if *text* asks to switch JARVIS's brain."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    for pattern in _SWITCH_PATTERNS:
        m = pattern.match(cleaned)
        if m:
            target = m.group(1).strip().strip("\"'")
            if target:
                return target
    return None


# ---------------------------------------------------------------------------
# Secure Key Store (~/.openjarvis/cloud-keys.env — compatible with OpenJarvis)
# ---------------------------------------------------------------------------


def _cloud_env_path() -> Path:
    return get_config_dir() / "cloud-keys.env"


def load_saved_keys() -> Dict[str, str]:
    """Read saved provider keys from ``~/.openjarvis/cloud-keys.env`` + process env."""
    keys: Dict[str, str] = {}
    env_file = _cloud_env_path()
    if env_file.exists():
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                val = v.strip().strip('"').strip("'")
                if k.strip() and val:
                    keys[k.strip()] = val
        except OSError:
            pass
    for env_name in list(PROVIDER_ENV_KEYS.values()) + [
        "GOOGLE_API_KEY",
        "OLLAMA_HOST",
    ]:
        val = os.environ.get(env_name, "").strip()
        if val:
            keys[env_name] = val
    return keys


def save_provider_secret(env_var: str, value: str) -> None:
    """Persist an API key or host setting to ``cloud-keys.env`` with chmod 0600."""
    env_var = env_var.strip()
    value = value.strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", env_var):
        raise ValueError(f"Invalid environment variable name: {env_var!r}")

    env_file = _cloud_env_path()
    env_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            existing[k.strip()] = v.strip().strip('"').strip("'")

    if value:
        existing[env_var] = value
        os.environ[env_var] = value
    else:
        existing.pop(env_var, None)
        os.environ.pop(env_var, None)

    lines = ["# OpenJarvis & My-JARVIS provider credentials (chmod 600)"]
    for k, v in sorted(existing.items()):
        safe_v = v.replace("\n", "").replace("\r", "")
        lines.append(f'{k}="{safe_v}"')
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "••••"
    return f"{secret[:4]}••••{secret[-4:]}"


# ---------------------------------------------------------------------------
# BrainRouter
# ---------------------------------------------------------------------------


class BrainRouter:
    """Manages JARVIS brain profiles, live health checks, and engine instantiation."""

    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path or (get_config_dir() / "my_jarvis_brains.json")
        self._brains: Dict[str, BrainProfile] = {
            b.id: BrainProfile(**asdict(b)) for b in DEFAULT_BRAINS
        }
        self._engine_cache: Dict[str, InferenceEngine] = {}
        self._active_id: str = "builtin-offline"
        self._ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        self._load_state()
        self._sync_keys_into_env()
        self._select_initial_brain()

    def _sync_keys_into_env(self) -> None:
        saved = load_saved_keys()
        for k, v in saved.items():
            if v and not os.environ.get(k):
                os.environ[k] = v
        if saved.get("OLLAMA_HOST"):
            self._ollama_host = saved["OLLAMA_HOST"]

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data.get("ollama_host"), str) and data["ollama_host"].strip():
            self._ollama_host = data["ollama_host"].strip()
        for raw in data.get("custom_brains", []):
            if isinstance(raw, dict) and raw.get("id") and raw.get("model"):
                try:
                    bp = BrainProfile(
                        id=str(raw["id"]),
                        name=str(raw.get("name", raw["id"])),
                        provider=str(raw.get("provider", "ollama")),
                        model=str(raw["model"]),
                        description=str(raw.get("description", "Custom brain profile")),
                        tier=str(raw.get("tier", "local")),
                        aliases=list(raw.get("aliases", [])),
                        env_var=str(raw.get("env_var", "")),
                        host=str(raw.get("host", "")),
                        cost_per_1m_in=float(raw.get("cost_per_1m_in", 0.0)),
                        cost_per_1m_out=float(raw.get("cost_per_1m_out", 0.0)),
                    )
                    self._brains[bp.id] = bp
                except Exception:
                    continue
        preferred = data.get("active_brain_id")
        if isinstance(preferred, str) and preferred in self._brains:
            self._active_id = preferred

    def _save_state(self) -> None:
        default_ids = {b.id for b in DEFAULT_BRAINS}
        custom = [
            b.to_dict() for bid, b in self._brains.items() if bid not in default_ids
        ]
        payload = {
            "active_brain_id": self._active_id,
            "ollama_host": self._ollama_host,
            "custom_brains": custom,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not persist brain state: %s", exc)

    def _select_initial_brain(self) -> None:
        """Pick the best ready brain on startup if active brain isn't explicitly ready."""
        if self._active_id != "builtin-offline" and self.is_brain_configured(
            self._brains[self._active_id]
        ):
            return
        # Prefer running local Ollama -> Venice -> Claude -> DeepSeek -> OpenAI -> Gemini -> Offline
        ollama_live, _ = self._probe_ollama_live()
        if ollama_live:
            self._active_id = "local-qwen"
            return
        for candidate in (
            "venice-deepseek",
            "claude-sonnet",
            "deepseek-cloud",
            "gpt-5",
            "gemini-flash",
            "openrouter-auto",
        ):
            if candidate in self._brains and self.is_brain_configured(
                self._brains[candidate]
            ):
                self._active_id = candidate
                return
        self._active_id = "builtin-offline"

    @property
    def active_brain(self) -> BrainProfile:
        return self._brains[self._active_id]

    @property
    def ollama_host(self) -> str:
        return self._ollama_host

    def _probe_ollama_live(self) -> tuple[bool, List[str]]:
        base = self._ollama_host.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        try:
            resp = httpx.get(f"{base}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                models = [
                    str(m.get("name", ""))
                    for m in resp.json().get("models", [])
                    if isinstance(m, dict) and m.get("name")
                ]
                return True, models
        except Exception:
            pass
        return False, []

    def is_brain_configured(self, brain: BrainProfile) -> bool:
        if brain.provider == "jarvis_offline":
            return True
        if brain.provider == "ollama":
            live, _ = self._probe_ollama_live()
            return live
        if brain.env_var:
            keys = load_saved_keys()
            if brain.env_var == "GEMINI_API_KEY" and keys.get("GOOGLE_API_KEY"):
                return True
            return bool(keys.get(brain.env_var))
        return True

    def list_brains(self, *, probe_live: bool = True) -> List[Dict[str, Any]]:
        """Return all registered brains with live availability metadata."""
        keys = load_saved_keys()
        ollama_live, ollama_models = (
            self._probe_ollama_live() if probe_live else (False, [])
        )
        items: List[Dict[str, Any]] = []
        for brain in self._brains.values():
            ready = False
            status_note = ""
            if brain.provider == "jarvis_offline":
                ready = True
                status_note = "Always ready"
            elif brain.provider == "ollama":
                ready = ollama_live
                if ollama_live:
                    installed = any(
                        m == brain.model
                        or m.startswith(f"{brain.model.split(':')[0]}:")
                        for m in ollama_models
                    )
                    status_note = (
                        "Ollama live · model installed"
                        if installed
                        else f"Ollama live · run `ollama pull {brain.model}`"
                    )
                else:
                    status_note = f"Ollama not running at {self._ollama_host}"
            else:
                has_key = bool(keys.get(brain.env_var)) or (
                    brain.env_var == "GEMINI_API_KEY"
                    and bool(keys.get("GOOGLE_API_KEY"))
                )
                ready = has_key
                status_note = (
                    f"{brain.env_var} configured ({mask_secret(keys.get(brain.env_var) or keys.get('GOOGLE_API_KEY', ''))})"
                    if has_key
                    else f"Set {brain.env_var} in Brains tab"
                )
            d = brain.to_dict()
            d["active"] = brain.id == self._active_id
            d["ready"] = ready
            d["status_note"] = status_note
            items.append(d)
        return items

    def resolve_brain(self, query: str) -> Optional[BrainProfile]:
        """Fuzzy-resolve a brain by ID, display name, model ID, or voice alias."""
        q = (query or "").strip().lower()
        if not q:
            return None
        # 1. Exact ID match
        if q in self._brains:
            return self._brains[q]
        # 2. Exact alias or model match
        for brain in self._brains.values():
            if q == brain.name.lower() or q == brain.model.lower():
                return brain
            if any(q == alias.lower() for alias in brain.aliases):
                return brain
        # 3. Substring match on ID, aliases, name, model
        for brain in self._brains.values():
            haystack = " ".join(
                [brain.id, brain.name, brain.model, brain.provider, *brain.aliases]
            ).lower()
            if q in haystack:
                return brain
        # 4. Token overlap match (great for noisy voice transcriptions!)
        q_tokens = set(re.findall(r"[a-z0-9]+", q))
        best: tuple[int, Optional[BrainProfile]] = (0, None)
        for brain in self._brains.values():
            b_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    " ".join(
                        [brain.id, brain.name, brain.model, *brain.aliases]
                    ).lower(),
                )
            )
            overlap = len(q_tokens & b_tokens)
            if overlap > best[0]:
                best = (overlap, brain)
        return best[1] if best[0] > 0 else None

    def switch_brain(self, query: str) -> Dict[str, Any]:
        """Hot-swap the active brain profile."""
        brain = self.resolve_brain(query)
        if brain is None:
            known = ", ".join(self._brains.keys())
            return {
                "ok": False,
                "error": f"Unknown brain {query!r}. Available brains: {known}",
            }
        prev = self._active_id
        self._active_id = brain.id
        self._save_state()
        ready = self.is_brain_configured(brain)
        return {
            "ok": True,
            "previous_brain_id": prev,
            "active_brain": brain.to_dict(),
            "ready": ready,
            "note": (
                f"Switched active brain to **{brain.name}** (`{brain.model}` via `{brain.provider}`)."
                if ready
                else (
                    f"Switched active brain to **{brain.name}** (`{brain.model}`), "
                    f"but `{brain.env_var or 'Ollama'}` is not configured yet. "
                    "Add your key or start Ollama in the Brains tab to go live."
                )
            ),
        }

    def add_custom_brain(
        self,
        *,
        brain_id: str,
        name: str,
        provider: str,
        model: str,
        description: str = "",
        host: str = "",
        env_var: str = "",
    ) -> BrainProfile:
        clean_id = re.sub(r"[^a-z0-9_-]+", "-", brain_id.strip().lower()).strip("-")
        if not clean_id:
            raise ValueError("brain_id must contain alphanumeric characters.")
        provider_clean = provider.strip().lower()
        if provider_clean not in ("ollama", "venice", "cloud", "llamacpp", "lmstudio"):
            raise ValueError(f"Unsupported provider: {provider!r}")
        tier = (
            "local"
            if provider_clean in ("ollama", "llamacpp", "lmstudio")
            else ("uncensored" if provider_clean == "venice" else "frontier")
        )
        if not env_var and provider_clean == "venice":
            env_var = "VENICE_API_KEY"
        bp = BrainProfile(
            id=clean_id,
            name=name.strip() or model,
            provider=provider_clean,
            model=model.strip(),
            description=description.strip()
            or f"Custom {provider_clean} brain ({model})",
            tier=tier,
            aliases=[clean_id, name.strip().lower(), model.strip().lower()],
            env_var=env_var.strip(),
            host=host.strip(),
        )
        self._brains[bp.id] = bp
        self._engine_cache.pop(bp.id, None)
        self._save_state()
        return bp

    def configure_provider(
        self,
        *,
        provider: str,
        api_key: str = "",
        host: str = "",
    ) -> Dict[str, Any]:
        """Update a provider's API key or endpoint host at runtime and persist it."""
        prov = provider.strip().lower()
        if prov == "ollama":
            if host:
                self._ollama_host = host.strip().rstrip("/")
                save_provider_secret("OLLAMA_HOST", self._ollama_host)
            self._engine_cache.clear()
            self._save_state()
            live, models = self._probe_ollama_live()
            return {
                "ok": True,
                "provider": "ollama",
                "host": self._ollama_host,
                "live": live,
                "models": models,
            }
        env_var = PROVIDER_ENV_KEYS.get(prov)
        if not env_var:
            raise ValueError(f"Unknown provider: {provider!r}")
        save_provider_secret(env_var, api_key)
        # Clear cached engine instances so the new credential takes effect immediately.
        self._engine_cache.clear()
        return {
            "ok": True,
            "provider": prov,
            "env_var": env_var,
            "configured": bool(api_key.strip()),
            "masked": mask_secret(api_key.strip()),
        }

    def get_provider_status(self) -> List[Dict[str, Any]]:
        """Return safe, masked status of every supported provider."""
        keys = load_saved_keys()
        ollama_live, ollama_models = self._probe_ollama_live()
        out = [
            {
                "provider": "ollama",
                "label": "Ollama (Local & Free)",
                "type": "local",
                "host": self._ollama_host,
                "configured": ollama_live,
                "status": f"Online ({len(ollama_models)} models)"
                if ollama_live
                else "Offline",
                "models": ollama_models,
            }
        ]
        meta = {
            "venice": ("Venice AI (Uncensored / Private)", "uncensored"),
            "anthropic": ("Anthropic Claude", "frontier"),
            "deepseek": ("DeepSeek Direct API", "frontier"),
            "openai": ("OpenAI GPT", "frontier"),
            "gemini": ("Google Gemini", "frontier"),
            "openrouter": ("OpenRouter (200+ Models)", "frontier"),
        }
        for prov, env_var in PROVIDER_ENV_KEYS.items():
            val = keys.get(env_var, "")
            if prov == "gemini" and not val:
                val = keys.get("GOOGLE_API_KEY", "")
            label, kind = meta[prov]
            out.append(
                {
                    "provider": prov,
                    "label": label,
                    "type": kind,
                    "env_var": env_var,
                    "configured": bool(val),
                    "masked": mask_secret(val),
                    "status": "Key Configured" if val else "Not Set",
                }
            )
        return out

    def get_engine_for(self, brain: Optional[BrainProfile] = None) -> InferenceEngine:
        """Instantiate or retrieve the OpenJarvis ``InferenceEngine`` for *brain*."""
        target = brain or self.active_brain
        cached = self._engine_cache.get(target.id)
        if cached is not None:
            return cached

        # Ensure OpenJarvis core engines are registered
        import openjarvis.engine  # noqa: F401

        if target.provider == "jarvis_offline":
            eng: InferenceEngine = BuiltinOfflineEngine()
        elif target.provider == "venice":
            eng = VeniceEngine(host=target.host or None)
        elif target.provider == "ollama":
            from openjarvis.engine.ollama import OllamaEngine

            eng = OllamaEngine(host=target.host or self._ollama_host)
        elif target.provider == "cloud":
            from openjarvis.engine.cloud import CloudEngine

            eng = CloudEngine()
        elif target.provider in ("llamacpp", "lmstudio"):
            cls = EngineRegistry.get(target.provider)
            eng = cls(host=target.host or None)
        else:
            eng = BuiltinOfflineEngine()

        self._engine_cache[target.id] = eng
        return eng


__all__ = [
    "BrainProfile",
    "BrainRouter",
    "BuiltinOfflineEngine",
    "DEFAULT_BRAINS",
    "PROVIDER_ENV_KEYS",
    "detect_brain_switch_intent",
    "load_saved_keys",
    "mask_secret",
    "save_provider_secret",
]
