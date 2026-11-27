#!/usr/bin/env bash
set -euo pipefail

# One-command setup for My-JARVIS built on top of Stanford OpenJarvis.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Syncing Stanford OpenJarvis + My-JARVIS dependencies via uv..."
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra server --extra inference-cloud

echo ""
echo "==> Running My-JARVIS diagnostic doctor..."
uv run my-jarvis doctor

echo ""
echo "==> Setup complete!"
echo "Start the JARVIS Command Center HUD + OpenJarvis API server:"
echo "    uv run my-jarvis serve --port 8765"
echo ""
echo "Or launch the interactive terminal REPL:"
echo "    uv run my-jarvis chat"
