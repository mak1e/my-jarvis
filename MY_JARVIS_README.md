# My-JARVIS (Built on Stanford's OpenJarvis)

A custom, voice-enabled, multi-brain personal AI built directly on top of Stanford SAIL's [OpenJarvis](https://github.com/open-jarvis/OpenJarvis) (`src/openjarvis/my_jarvis/`).

## What's Built on Top of Stanford OpenJarvis

1. **Native Venice AI Engine (`VeniceEngine`)** — Registered into OpenJarvis's `EngineRegistry` (`venice`), targeting `https://api.venice.ai/api/v1` for private / uncensored open-weight inference (DeepSeek, Venice Uncensored, Qwen, Llama) with `include_venice_system_prompt: false`.
2. **Hot-Swappable Multi-Brain Router (`BrainRouter`)** — Switch JARVIS's brain mid-conversation via voice (*"Hey Jarvis, switch your brain to DeepSeek"*), `/brain <name>`, OpenJarvis tool calls, or 1-click HUD toggle across **Local Ollama**, **Venice AI**, **Claude Opus/Sonnet**, **DeepSeek Direct**, **GPT-5 Mini**, **Gemini 2.5 Flash**, **OpenRouter**, and an always-ready **Offline Core**.
3. **Executive Morning Briefing (`MorningBriefingService`)** — Combines today's Google Calendar events, recent Google Drive files, local weather, top Hacker News headlines, and your OpenJarvis `LocalFactStore` priorities into both structured HUD cards and a spoken script (`my-jarvis briefing` or say *"give me my morning briefing"*).
4. **Multi-Platform Messaging Channels (`JarvisChannelManager`)** — Bridges OpenJarvis's native **Telegram**, **Discord**, **Slack**, and **Webhook** (`POST /api/jarvis/channels/webhook/inbound` for Apple Shortcuts / n8n / Zapier / WhatsApp relays) directly into `MyJarvisRuntime` so you can talk to JARVIS and switch brains remotely.
5. **MCP Server Hub (`JarvisMCPHub`)** — Connects Stdio and Streamable HTTP Model Context Protocol servers via OpenJarvis's `MCPClient` + `MCPToolProvider` and automatically exposes every discovered tool inside `OrchestratorAgent` (with starter templates for Filesystem, Web Fetch, GitHub MCP, and Playwright).
6. **Neural Voice Pipeline (`JarvisSpeechHub`)** — Upgrades smoothly from zero-dependency Browser Web Speech API (Push-to-Talk & Hands-Free *"Hey Jarvis"* wake mode) to OpenJarvis's neural TTS/STT backends (**Kokoro local TTS**, **Faster-Whisper local STT**, **OpenAI TTS/Whisper**, **ElevenLabs**, **Cartesia**, **Deepgram**).
7. **Hardware / VRAM Advisor & Hugging Face GGUF Explorer** — Measures CPU/RAM/GPU VRAM, scores fitting local open-weight models, searches Hugging Face Hub for GGUF models, and outputs copyable `ollama pull` / `ollama run hf.co/...` commands.
8. **Defensive Self-Auditor** — Authorized passive HTTP security header & TLS posture grader (`A+`–`F` with SSRF guard) and local codebase secret/hygiene scanner.

## Clone & Run on Your Machine

```bash
git clone -b custom/my-jarvis https://github.com/mak1e/my-jarvis.git
cd my-jarvis
./scripts/setup_my_jarvis.sh

# Optional: free local LLM via Ollama
ollama serve &
ollama pull qwen3:8b

# Launch the JARVIS Command Center HUD + OpenJarvis /v1/* API
uv run my-jarvis serve --port 8765
```

### Optional OpenJarvis Extras

```bash
uv sync --extra voice             # Local Kokoro neural TTS
uv sync --extra speech            # Local Faster-Whisper STT
uv sync --extra channel-discord   # Discord bot gateway
uv sync --extra channel-slack     # Slack Socket Mode bot
```

### CLI Commands

```bash
uv run my-jarvis serve --port 8765   # Launch web HUD + OpenJarvis /v1/* API
uv run my-jarvis chat                # Interactive terminal REPL with /brain hot-swap
uv run my-jarvis briefing            # Generate executive Morning Briefing
uv run my-jarvis brains              # List all brains & live provider readiness
uv run my-jarvis channels            # Inspect Telegram/Discord/Slack/Webhook status
uv run my-jarvis mcp                 # Inspect registered MCP servers & discovered tools
uv run my-jarvis voice               # Inspect Kokoro/OpenAI/ElevenLabs/Faster-Whisper status
uv run my-jarvis hardware            # Inspect RAM/VRAM & best-fit local GGUF models
uv run my-jarvis doctor              # Run health checks & setup fix tips
```
