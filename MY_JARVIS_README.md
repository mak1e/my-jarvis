# My-JARVIS (Built on Stanford's OpenJarvis)

A custom, voice-enabled, multi-brain personal AI built directly on top of Stanford SAIL's [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) (`src/openjarvis/my_jarvis/`).

## What We Built on Top of OpenJarvis

1. **Native Venice AI Engine (`VeniceEngine`)** — Registered into OpenJarvis's `EngineRegistry` (`venice`), targeting `https://api.venice.ai/api/v1` for private / uncensored open-weight cloud inference (DeepSeek, Venice Uncensored, Qwen, Llama) with `include_venice_system_prompt: false` support.
2. **Hot-Swappable Multi-Brain Router (`BrainRouter`)** — Switch JARVIS's brain mid-conversation via voice (*"Jarvis, switch your brain to DeepSeek"*), chat command (`/brain claude-opus`), OpenJarvis tool call (`switch_brain`), or 1-click HUD card across:
   - **Local Ollama** (`qwen3:8b`, `qwen3:32b`, `deepseek-r1:8b`, or any pulled Hugging Face GGUF)
   - **Venice AI** (`deepseek-r1-671b`, `venice-uncensored`)
   - **Frontier Cloud via OpenJarvis `CloudEngine`** (`claude-opus-4-6`, `claude-sonnet-4-6`, `deepseek-v4-flash`, `gpt-5-mini`, `gemini-2.5-flash`, `openrouter/auto`)
   - **Built-in Offline Core** (`jarvis-offline-core`) so local tools, memory, hardware checks, and defensive audits work immediately with zero API keys or 500 crashes.
3. **JARVIS Command Center HUD (`my-jarvis serve`)** — Dark aerospace telemetry web app mounted on top of OpenJarvis's FastAPI server (`create_app`), featuring:
   - Canvas-rendered acoustic arc-reactor visualizer (cyan for local/frontier, amber-gold for Venice/uncensored)
   - Browser Web Speech API **Push-to-Talk & Hands-Free Wake Word ("Hey Jarvis")** + Speech Synthesis TTS
   - Live Provider & API Key Manager (stored locally in `~/.openjarvis/cloud-keys.env` with `chmod 600`)
   - **Persistent Memory Vault** backed by OpenJarvis's `LocalFactStore` (retained across every brain switch)
   - **Hardware & Hugging Face GGUF Model Advisor** (measures CPU/RAM/GPU VRAM and outputs exact `ollama pull` / `ollama run hf.co/...` commands)
   - **Defensive Self-Auditor** (passive HTTP security header & TLS posture grader for sites you own + local codebase secret/hygiene scan)
   - **Setup & Doctor Diagnostics** (`my-jarvis doctor`)

## Quick Start on Your Machine

```bash
# 1. Sync Python dependencies (Python 3.11-3.13 + uv)
./scripts/setup_my_jarvis.sh

# 2. (Optional, free & local) Install Ollama and pull a local model
ollama serve &
ollama pull qwen3:8b

# 3. Launch the JARVIS Command Center HUD + OpenAI-compatible OpenJarvis API
uv run my-jarvis serve --port 8765
```

### CLI Commands

```bash
uv run my-jarvis serve --port 8765   # Launch web HUD + OpenJarvis /v1/* API
uv run my-jarvis chat                # Interactive terminal REPL with /brain hot-swap
uv run my-jarvis brains              # List all brains & live provider readiness
uv run my-jarvis hardware            # Inspect RAM/VRAM & best-fit local GGUF models
uv run my-jarvis doctor              # Run health checks & setup fix tips
```
