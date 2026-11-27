"""MCP (Model Context Protocol) Server Hub built on OpenJarvis's native MCP client.

Lets users register external MCP servers (Stdio or Streamable HTTP), discover
their tools via ``openjarvis.tools.mcp_adapter.MCPToolProvider``, inspect live
tool schemas, and automatically inject every discovered MCP tool into
``MyJarvisRuntime``'s OpenJarvis ``OrchestratorAgent``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjarvis.core.paths import get_config_dir
from openjarvis.tools._stubs import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSpec:
    id: str
    name: str
    transport: str  # "http" | "stdio"
    url: str = ""
    token: str = ""
    command: str = ""
    args: List[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["has_token"] = bool(self.token)
        d.pop("token", None)
        return d


STARTER_MCP_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "filesystem",
        "name": "Local Filesystem MCP",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
        "description": "Read, search, and manage files inside an authorized local folder.",
    },
    {
        "id": "fetch",
        "name": "Web Fetch MCP",
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Fetch web pages and convert HTML cleanly into Markdown for JARVIS.",
    },
    {
        "id": "github-mcp",
        "name": "GitHub Official MCP",
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "description": "Query PRs, issues, code search, and releases via Streamable HTTP MCP.",
    },
    {
        "id": "playwright",
        "name": "Playwright Browser MCP",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "description": "Give JARVIS live browser navigation, screenshots, and DOM interaction.",
    },
]


class JarvisMCPHub:
    """Persists MCP server specs and manages live OpenJarvis ``MCPClient`` sessions."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._path = config_path or (get_config_dir() / "my_jarvis_mcp.json")
        self._servers: Dict[str, MCPServerSpec] = {}
        self._clients: Dict[str, Any] = {}
        self._tools: Dict[str, List[BaseTool]] = {}
        self._errors: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and item.get("id"):
                        spec = MCPServerSpec(
                            id=str(item["id"]),
                            name=str(item.get("name", item["id"])),
                            transport=str(item.get("transport", "http")),
                            url=str(item.get("url", "")),
                            token=str(item.get("token", "")),
                            command=str(item.get("command", "")),
                            args=list(item.get("args", [])),
                            enabled=bool(item.get("enabled", True)),
                            description=str(item.get("description", "")),
                        )
                        self._servers[spec.id] = spec
        except Exception as exc:
            logger.debug("Failed loading MCP server config: %s", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(s) for s in self._servers.values()]
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def add_server(
        self,
        *,
        server_id: str,
        name: str,
        transport: str,
        url: str = "",
        token: str = "",
        command: str = "",
        args: Optional[List[str]] = None,
        description: str = "",
    ) -> MCPServerSpec:
        clean_id = re.sub(r"[^a-z0-9_-]+", "-", server_id.strip().lower()).strip("-")
        if not clean_id:
            raise ValueError("server_id must contain alphanumeric characters.")
        t = transport.strip().lower()
        if t not in ("http", "stdio"):
            raise ValueError("transport must be 'http' or 'stdio'.")
        if t == "http" and not url.strip():
            raise ValueError("HTTP MCP server requires a non-empty 'url'.")
        if t == "stdio" and not command.strip():
            raise ValueError("Stdio MCP server requires a non-empty 'command'.")

        spec = MCPServerSpec(
            id=clean_id,
            name=name.strip() or clean_id,
            transport=t,
            url=url.strip(),
            token=token.strip(),
            command=command.strip(),
            args=[str(a) for a in (args or []) if str(a).strip()],
            description=description.strip(),
        )
        self._servers[spec.id] = spec
        self._errors.pop(spec.id, None)
        self._save()
        return spec

    def remove_server(self, server_id: str) -> Dict[str, Any]:
        sid = server_id.strip().lower()
        self.disconnect(sid)
        removed = self._servers.pop(sid, None)
        self._errors.pop(sid, None)
        self._save()
        return {"ok": removed is not None, "id": sid}

    def connect(
        self,
        server_id: str,
        *,
        client_factory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Connect to a configured MCP server and discover its tools via OpenJarvis."""
        sid = server_id.strip().lower()
        spec = self._servers.get(sid)
        if spec is None:
            raise ValueError(f"Unknown MCP server: {server_id!r}")

        self.disconnect(sid)

        from openjarvis.mcp.client import MCPClient
        from openjarvis.mcp.transport import StdioTransport, StreamableHTTPTransport
        from openjarvis.tools.mcp_adapter import MCPToolProvider

        try:
            if client_factory is not None:
                client = client_factory(spec)
            elif spec.transport == "http":
                transport = StreamableHTTPTransport(
                    url=spec.url,
                    token=spec.token or None,
                )
                client = MCPClient(transport)
            else:
                transport = StdioTransport(command=[spec.command, *spec.args])
                client = MCPClient(transport)

            client.initialize()
            provider = MCPToolProvider(client)
            discovered = provider.discover()

            self._clients[sid] = client
            self._tools[sid] = discovered
            self._errors.pop(sid, None)
            return {
                "ok": True,
                "id": sid,
                "connected": True,
                "tools": [
                    {
                        "name": t.spec.name,
                        "description": t.spec.description,
                    }
                    for t in discovered
                ],
            }
        except Exception as exc:
            self._errors[sid] = str(exc)
            logger.warning("MCP server %s failed to connect: %s", sid, exc)
            return {
                "ok": False,
                "id": sid,
                "connected": False,
                "error": str(exc),
            }

    def disconnect(self, server_id: str) -> Dict[str, Any]:
        sid = server_id.strip().lower()
        client = self._clients.pop(sid, None)
        self._tools.pop(sid, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        return {"ok": True, "id": sid, "connected": False}

    def close_all(self) -> None:
        for sid in list(self._clients.keys()):
            self.disconnect(sid)

    def get_active_tools(self) -> List[BaseTool]:
        """Return all live ``MCPToolAdapter`` BaseTools for ``OrchestratorAgent``."""
        tools: List[BaseTool] = []
        for server_tools in self._tools.values():
            tools.extend(server_tools)
        return tools

    def list_servers(self) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for sid, spec in self._servers.items():
            d = spec.to_public_dict()
            live_tools = self._tools.get(sid, [])
            d["connected"] = sid in self._clients
            d["tool_count"] = len(live_tools)
            d["tools"] = [
                {"name": t.spec.name, "description": t.spec.description}
                for t in live_tools
            ]
            d["error"] = self._errors.get(sid, "")
            rows.append(d)
        return {
            "servers": rows,
            "total_mcp_tools": len(self.get_active_tools()),
            "templates": STARTER_MCP_TEMPLATES,
        }


__all__ = ["JarvisMCPHub", "MCPServerSpec", "STARTER_MCP_TEMPLATES"]
