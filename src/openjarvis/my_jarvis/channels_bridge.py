"""Multi-platform messaging channel manager built on OpenJarvis channels.

Bridges OpenJarvis's native ``TelegramChannel``, ``DiscordChannel``,
``SlackChannel``, and inbound webhooks straight into ``MyJarvisRuntime`` so you
can chat with JARVIS, hot-swap brains (``/brain deepseek``), trigger morning
briefings, and save memories from your phone or team workspace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from openjarvis.channels._stubs import BaseChannel, ChannelMessage, ChannelStatus
from openjarvis.core.events import EventBus
from openjarvis.my_jarvis.brains import (
    load_saved_keys,
    mask_secret,
    save_provider_secret,
)

logger = logging.getLogger(__name__)

CHANNEL_DEFS: Dict[str, Dict[str, Any]] = {
    "telegram": {
        "label": "Telegram Bot",
        "token_env": "TELEGRAM_BOT_TOKEN",
        "extra_env": "TELEGRAM_ALLOWED_CHAT_IDS",
        "extra_label": "Allowed Chat IDs (comma-separated, optional)",
        "install_hint": "Included in OpenJarvis core (python-telegram-bot)",
    },
    "discord": {
        "label": "Discord Bot",
        "token_env": "DISCORD_BOT_TOKEN",
        "extra_env": "",
        "extra_label": "",
        "install_hint": "uv sync --extra channel-discord",
    },
    "slack": {
        "label": "Slack App (Socket Mode)",
        "token_env": "SLACK_BOT_TOKEN",
        "extra_env": "SLACK_APP_TOKEN",
        "extra_label": "Slack App-Level Token (xapp-... for Socket Mode)",
        "install_hint": "uv sync --extra channel-slack",
    },
    "webhook": {
        "label": "Generic Webhook (Shortcuts / n8n / Zapier / WhatsApp Relay)",
        "token_env": "JARVIS_WEBHOOK_SECRET",
        "extra_env": "",
        "extra_label": "",
        "install_hint": "Built-in — POST /api/jarvis/channels/webhook/inbound",
    },
}


@dataclass
class ChannelActivity:
    timestamp: float
    channel: str
    sender: str
    inbound: str
    reply_preview: str
    brain_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JarvisChannelManager:
    """Lifecycle and routing manager for My-JARVIS messaging channels."""

    def __init__(
        self,
        *,
        chat_callback: Callable[[str], Dict[str, Any]],
        bus: Optional[EventBus] = None,
    ) -> None:
        self._chat_callback = chat_callback
        self._bus = bus
        self._channels: Dict[str, BaseChannel] = {}
        self._activity: List[ChannelActivity] = []

    def _package_available(self, channel_id: str) -> bool:
        if channel_id == "webhook":
            return True
        if channel_id == "telegram":
            try:
                import telegram  # noqa: F401

                return True
            except ImportError:
                return False
        if channel_id == "discord":
            try:
                import discord  # noqa: F401

                return True
            except ImportError:
                return False
        if channel_id == "slack":
            try:
                import slack_sdk  # noqa: F401

                return True
            except ImportError:
                return False
        return False

    def configure_channel(
        self,
        channel_id: str,
        *,
        token: str = "",
        extra: str = "",
    ) -> Dict[str, Any]:
        cid = channel_id.strip().lower()
        meta = CHANNEL_DEFS.get(cid)
        if not meta:
            raise ValueError(f"Unsupported channel: {channel_id!r}")
        if token is not None:
            save_provider_secret(meta["token_env"], token.strip())
        if meta["extra_env"] and extra is not None:
            save_provider_secret(meta["extra_env"], extra.strip())
        return {
            "ok": True,
            "channel": cid,
            "configured": bool(token.strip()),
            "masked": mask_secret(token.strip()),
        }

    def _handle_inbound_message(self, msg: ChannelMessage) -> Optional[str]:
        """Bridge incoming OpenJarvis ``ChannelMessage`` into ``MyJarvisRuntime``."""
        text = (msg.content or "").strip()
        if not text:
            return None
        try:
            result = self._chat_callback(text)
            reply_obj = result.get("reply") or {}
            reply_text = str(reply_obj.get("content") or "")
            brain_id = str(reply_obj.get("brain_id") or "")
            self._record_activity(
                channel=msg.channel,
                sender=msg.sender or "unknown",
                inbound=text,
                reply_preview=reply_text[:200],
                brain_id=brain_id,
            )
            return reply_text
        except Exception as exc:
            logger.exception("Channel %s message handling failed", msg.channel)
            return f"JARVIS encountered an error processing your message: {exc}"

    def _record_activity(
        self,
        *,
        channel: str,
        sender: str,
        inbound: str,
        reply_preview: str,
        brain_id: str,
    ) -> None:
        self._activity.append(
            ChannelActivity(
                timestamp=time.time(),
                channel=channel,
                sender=sender,
                inbound=inbound[:200],
                reply_preview=reply_preview,
                brain_id=brain_id,
            )
        )
        if len(self._activity) > 50:
            self._activity = self._activity[-50:]

    def connect(self, channel_id: str) -> Dict[str, Any]:
        """Start listening on an OpenJarvis channel adapter."""
        cid = channel_id.strip().lower()
        meta = CHANNEL_DEFS.get(cid)
        if not meta:
            raise ValueError(f"Unsupported channel: {channel_id!r}")

        if cid == "webhook":
            return {
                "ok": True,
                "channel": "webhook",
                "status": ChannelStatus.CONNECTED.value,
                "note": "Webhook endpoint is always active at POST /api/jarvis/channels/webhook/inbound.",
            }

        keys = load_saved_keys()
        token = keys.get(meta["token_env"], "").strip()
        if not token:
            return {
                "ok": False,
                "channel": cid,
                "error": f"Set {meta['token_env']} before connecting {meta['label']}.",
            }
        if not self._package_available(cid):
            return {
                "ok": False,
                "channel": cid,
                "error": f"Optional dependency missing. Install with: {meta['install_hint']}",
            }

        # Disconnect any prior instance cleanly
        if cid in self._channels:
            try:
                self._channels[cid].disconnect()
            except Exception:
                pass

        adapter: BaseChannel
        if cid == "telegram":
            from openjarvis.channels.telegram import TelegramChannel

            allowed = keys.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
            adapter = TelegramChannel(
                bot_token=token,
                allowed_chat_ids=allowed,
                bus=self._bus,
            )
        elif cid == "discord":
            from openjarvis.channels.discord_channel import DiscordChannel

            adapter = DiscordChannel(bot_token=token, bus=self._bus)
        elif cid == "slack":
            from openjarvis.channels.slack import SlackChannel

            app_token = keys.get("SLACK_APP_TOKEN", "")
            adapter = SlackChannel(
                bot_token=token,
                app_token=app_token,
                bus=self._bus,
            )
        else:
            raise ValueError(f"Unsupported channel: {cid!r}")

        adapter.on_message(self._handle_inbound_message)
        adapter.connect()
        self._channels[cid] = adapter
        return {
            "ok": adapter.status()
            in (ChannelStatus.CONNECTED, ChannelStatus.CONNECTING),
            "channel": cid,
            "status": adapter.status().value,
        }

    def disconnect(self, channel_id: str) -> Dict[str, Any]:
        cid = channel_id.strip().lower()
        adapter = self._channels.pop(cid, None)
        if adapter is not None:
            try:
                adapter.disconnect()
            except Exception as exc:
                logger.debug("Channel %s disconnect error: %s", cid, exc)
        return {"ok": True, "channel": cid, "status": ChannelStatus.DISCONNECTED.value}

    def disconnect_all(self) -> None:
        for cid in list(self._channels.keys()):
            self.disconnect(cid)

    def handle_webhook_inbound(
        self,
        *,
        sender: str,
        content: str,
        channel_name: str = "webhook",
        secret: str = "",
    ) -> Dict[str, Any]:
        """Process an inbound webhook message (supports Shortcuts, n8n, Zapier, WhatsApp relays)."""
        keys = load_saved_keys()
        expected = keys.get("JARVIS_WEBHOOK_SECRET", "").strip()
        if expected and secret != expected:
            return {"ok": False, "error": "Invalid webhook secret.", "status_code": 401}

        cm = ChannelMessage(
            channel=channel_name or "webhook",
            sender=sender or "webhook-user",
            content=content,
            conversation_id=sender or "webhook",
        )
        reply = self._handle_inbound_message(cm) or ""
        return {
            "ok": True,
            "channel": cm.channel,
            "sender": cm.sender,
            "reply": reply,
        }

    def list_channels(self) -> List[Dict[str, Any]]:
        keys = load_saved_keys()
        rows: List[Dict[str, Any]] = []
        for cid, meta in CHANNEL_DEFS.items():
            token = keys.get(meta["token_env"], "")
            extra_val = keys.get(meta["extra_env"], "") if meta["extra_env"] else ""
            adapter = self._channels.get(cid)
            if cid == "webhook":
                status = ChannelStatus.CONNECTED.value
            elif adapter is not None:
                status = adapter.status().value
            else:
                status = ChannelStatus.DISCONNECTED.value
            rows.append(
                {
                    "id": cid,
                    "label": meta["label"],
                    "token_env": meta["token_env"],
                    "extra_env": meta["extra_env"],
                    "extra_label": meta["extra_label"],
                    "configured": bool(token) if cid != "webhook" else True,
                    "secret_set": bool(token),
                    "masked_token": mask_secret(token),
                    "extra_configured": bool(extra_val),
                    "dependency_installed": self._package_available(cid),
                    "install_hint": meta["install_hint"],
                    "status": status,
                }
            )
        return rows

    def get_activity(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in reversed(self._activity[-20:])]


__all__ = ["CHANNEL_DEFS", "ChannelActivity", "JarvisChannelManager"]
