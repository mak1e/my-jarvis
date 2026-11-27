"""Neural speech-to-text & text-to-speech hub built on OpenJarvis speech backends.

Unifies OpenJarvis's ``TTSRegistry`` (Kokoro local neural TTS, OpenAI TTS,
ElevenLabs, Cartesia) and ``SpeechRegistry`` (Faster-Whisper local STT, OpenAI
Whisper, Deepgram) so the HUD and CLI can upgrade smoothly from zero-dependency
Browser Speech to studio-grade neural voices.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.my_jarvis.brains import (
    load_saved_keys,
    mask_secret,
    save_provider_secret,
)
from openjarvis.speech._tts_discovery import (
    BACKEND_DEFAULT_VOICE,
    TTS_BACKEND_ORDER,
    get_tts_backend,
)

logger = logging.getLogger(__name__)

TTS_META: Dict[str, Dict[str, Any]] = {
    "kokoro": {
        "label": "Kokoro Neural TTS (100% Local)",
        "type": "local",
        "env_var": "",
        "default_voice": BACKEND_DEFAULT_VOICE.get("kokoro", "bm_george"),
        "install_hint": "uv sync --extra voice   # plus espeak-ng",
    },
    "openai_tts": {
        "label": "OpenAI Neural TTS (onyx / alloy)",
        "type": "cloud",
        "env_var": "OPENAI_API_KEY",
        "default_voice": BACKEND_DEFAULT_VOICE.get("openai_tts", "onyx"),
        "install_hint": "Set OPENAI_API_KEY in Brains & Keys",
    },
    "elevenlabs": {
        "label": "ElevenLabs Voice (British 'George')",
        "type": "cloud",
        "env_var": "ELEVENLABS_API_KEY",
        "default_voice": BACKEND_DEFAULT_VOICE.get(
            "elevenlabs", "JBFqnCBsd6RMkjVDRZzb"
        ),
        "install_hint": "Set ELEVENLABS_API_KEY below",
    },
    "cartesia": {
        "label": "Cartesia Sonic (Low-Latency TTS)",
        "type": "cloud",
        "env_var": "CARTESIA_API_KEY",
        "default_voice": "",
        "install_hint": "Set CARTESIA_API_KEY below",
    },
}

STT_META: Dict[str, Dict[str, Any]] = {
    "faster-whisper": {
        "label": "Faster-Whisper (100% Local STT)",
        "type": "local",
        "env_var": "",
        "install_hint": "uv sync --extra speech",
    },
    "openai": {
        "label": "OpenAI Whisper API",
        "type": "cloud",
        "env_var": "OPENAI_API_KEY",
        "install_hint": "Set OPENAI_API_KEY in Brains & Keys",
    },
    "deepgram": {
        "label": "Deepgram Nova STT",
        "type": "cloud",
        "env_var": "DEEPGRAM_API_KEY",
        "install_hint": "uv sync --extra speech-deepgram + DEEPGRAM_API_KEY",
    },
}

SPEECH_SECRET_KEYS = {
    "elevenlabs": "ELEVENLABS_API_KEY",
    "cartesia": "CARTESIA_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


class JarvisSpeechHub:
    """Manages neural TTS and STT discovery, key storage, synthesis, and transcription."""

    def __init__(self) -> None:
        self._preferred_tts: str = "auto"

    def configure_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        prov = provider.strip().lower()
        env_var = SPEECH_SECRET_KEYS.get(prov)
        if not env_var:
            raise ValueError(f"Unsupported speech key provider: {provider!r}")
        save_provider_secret(env_var, api_key.strip())
        return {
            "ok": True,
            "provider": prov,
            "env_var": env_var,
            "configured": bool(api_key.strip()),
            "masked": mask_secret(api_key.strip()),
        }

    def status(self) -> Dict[str, Any]:
        """Inspect installed & healthy OpenJarvis TTS and STT backends."""
        import openjarvis.speech  # noqa: F401
        from openjarvis.core.registry import SpeechRegistry, TTSRegistry

        keys = load_saved_keys()
        tts_rows: List[Dict[str, Any]] = []
        any_tts_ready = False

        for key in TTS_BACKEND_ORDER:
            meta = TTS_META[key]
            registered = TTSRegistry.contains(key)
            healthy = False
            voices: List[str] = []
            if registered:
                try:
                    inst = TTSRegistry.get(key)()
                    healthy = bool(inst.health())
                    if healthy:
                        voices = inst.available_voices()[:12]
                except Exception:
                    healthy = False
            if healthy:
                any_tts_ready = True
            env_var = meta["env_var"]
            key_val = keys.get(env_var, "") if env_var else ""
            tts_rows.append(
                {
                    "id": key,
                    "label": meta["label"],
                    "type": meta["type"],
                    "registered": registered,
                    "ready": healthy,
                    "default_voice": meta["default_voice"],
                    "voices": voices,
                    "env_var": env_var,
                    "masked_key": mask_secret(key_val),
                    "install_hint": meta["install_hint"],
                }
            )

        stt_rows: List[Dict[str, Any]] = []
        any_stt_ready = False
        for key, meta in STT_META.items():
            registered = SpeechRegistry.contains(key)
            healthy = False
            if registered:
                try:
                    cls = SpeechRegistry.get(key)
                    if key == "faster-whisper":
                        # Check package importability without downloading weights
                        import faster_whisper  # noqa: F401

                        healthy = True
                    elif key == "openai":
                        healthy = bool(keys.get("OPENAI_API_KEY"))
                    elif key == "deepgram":
                        healthy = bool(keys.get("DEEPGRAM_API_KEY"))
                    else:
                        healthy = bool(cls().health())
                except Exception:
                    healthy = False
            if healthy:
                any_stt_ready = True
            env_var = meta["env_var"]
            key_val = keys.get(env_var, "") if env_var else ""
            stt_rows.append(
                {
                    "id": key,
                    "label": meta["label"],
                    "type": meta["type"],
                    "registered": registered,
                    "ready": healthy,
                    "env_var": env_var,
                    "masked_key": mask_secret(key_val),
                    "install_hint": meta["install_hint"],
                }
            )

        return {
            "neural_tts_ready": any_tts_ready,
            "neural_stt_ready": any_stt_ready,
            "preferred_tts": self._preferred_tts,
            "tts_backends": tts_rows,
            "stt_backends": stt_rows,
        }

    def synthesize(
        self,
        text: str,
        *,
        backend: str = "",
        voice_id: str = "",
        speed: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Synthesize *text* via the first healthy OpenJarvis neural TTS backend."""
        clean = (text or "").strip()
        if not clean:
            return None
        preferred = backend if backend and backend != "auto" else ""
        tts = get_tts_backend(preferred=preferred)
        if tts is None:
            return None
        resolved_voice = voice_id or BACKEND_DEFAULT_VOICE.get(
            getattr(tts, "backend_id", ""), ""
        )
        result = tts.synthesize(
            clean[:1200],
            voice_id=resolved_voice,
            speed=max(0.5, min(2.0, float(speed))),
        )
        media_type = (
            "audio/wav" if (result.format or "").lower() == "wav" else "audio/mpeg"
        )
        return {
            "audio": result.audio,
            "format": result.format,
            "media_type": media_type,
            "backend": getattr(tts, "backend_id", "unknown"),
            "voice_id": result.voice_id or resolved_voice,
        }

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.webm",
    ) -> Dict[str, Any]:
        """Transcribe uploaded audio via OpenJarvis's healthy STT backend."""
        from openjarvis.core.config import load_config
        from openjarvis.speech._discovery import get_speech_backend

        cfg = load_config()
        backend = get_speech_backend(cfg)
        if backend is None:
            return {
                "ok": False,
                "error": (
                    "No backend STT engine ready. Install local Faster-Whisper "
                    "(`uv sync --extra speech`) or set OPENAI_API_KEY."
                ),
            }
        suffix = Path(filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            res = backend.transcribe(Path(tmp.name))
            return {
                "ok": True,
                "text": getattr(res, "text", str(res)),
                "backend": getattr(backend, "backend_id", "unknown"),
            }


__all__ = ["JarvisSpeechHub", "SPEECH_SECRET_KEYS", "STT_META", "TTS_META"]
