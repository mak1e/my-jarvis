"""Executive Morning Briefing generator built on OpenJarvis connectors + Workspace CLIs.

Collects read-only personal productivity & world context:
- Today's Google Calendar events (via ``gcalendar`` CLI or OpenJarvis connector when ready)
- Recent Google Drive files (via ``gdrive`` CLI when ready)
- Current local Weather (via OpenWeatherMap or wttr.in JSON fallback)
- Top Hacker News / Tech Headlines (via OpenJarvis's HackerNews Firebase feed)
- Personal Priorities & Facts (from OpenJarvis's ``LocalFactStore``)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from openjarvis.core.paths import get_config_dir
from openjarvis.memory.store import LocalFactStore

logger = logging.getLogger(__name__)


@dataclass
class BriefingConfig:
    city: str = "San Francisco"
    include_calendar: bool = True
    include_drive: bool = True
    include_weather: bool = True
    include_news: bool = True
    news_limit: int = 5
    schedule_cron: str = "0 8 * * *"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BriefingReport:
    generated_at: str
    greeting: str
    city: str
    weather: Dict[str, Any] = field(default_factory=dict)
    calendar_events: List[Dict[str, Any]] = field(default_factory=list)
    calendar_source: str = "unavailable"
    drive_files: List[Dict[str, Any]] = field(default_factory=list)
    drive_source: str = "unavailable"
    news: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)
    markdown: str = ""
    spoken_script: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _time_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _fetch_weather(city: str) -> Dict[str, Any]:
    clean_city = (city or "San Francisco").strip()
    owm_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if owm_key:
        try:
            resp = httpx.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": clean_city, "appid": owm_key, "units": "imperial"},
                timeout=4.0,
            )
            if resp.status_code == 200:
                d = resp.json()
                return {
                    "available": True,
                    "source": "openweathermap",
                    "city": d.get("name", clean_city),
                    "temp_f": round(float(d["main"]["temp"]), 1),
                    "feels_like_f": round(float(d["main"]["feels_like"]), 1),
                    "condition": d["weather"][0]["description"].title(),
                    "humidity": d["main"].get("humidity"),
                }
        except Exception as exc:
            logger.debug("OpenWeatherMap fetch failed: %s", exc)

    # Free zero-key fallback via wttr.in
    try:
        resp = httpx.get(
            f"https://wttr.in/{clean_city}",
            params={"format": "j1"},
            headers={"User-Agent": "OpenJarvis-Briefing/1.0"},
            timeout=4.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            cur = (data.get("current_condition") or [{}])[0]
            desc_list = cur.get("weatherDesc") or [{}]
            return {
                "available": True,
                "source": "wttr.in",
                "city": clean_city,
                "temp_f": cur.get("temp_F", "?"),
                "temp_c": cur.get("temp_C", "?"),
                "feels_like_f": cur.get("FeelsLikeF", "?"),
                "condition": desc_list[0].get("value", "Clear"),
                "humidity": cur.get("humidity", "?"),
            }
    except Exception as exc:
        logger.debug("wttr.in weather fetch unavailable: %s", exc)

    return {
        "available": False,
        "source": "offline",
        "city": clean_city,
        "note": "Weather service unreachable in current network environment.",
    }


def _fetch_hackernews(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch top Hacker News tech headlines using OpenJarvis's HN Firebase feed."""
    stories: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=4.0) as client:
            ids_resp = client.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json"
            )
            if ids_resp.status_code != 200:
                return []
            ids = ids_resp.json()[: max(1, min(limit, 10))]
            for item_id in ids:
                item_resp = client.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
                )
                if item_resp.status_code == 200:
                    item = item_resp.json() or {}
                    if item.get("title"):
                        stories.append(
                            {
                                "title": str(item["title"]),
                                "score": int(item.get("score", 0) or 0),
                                "url": str(
                                    item.get("url")
                                    or f"https://news.ycombinator.com/item?id={item_id}"
                                ),
                            }
                        )
    except Exception as exc:
        logger.debug("Hacker News fetch unavailable: %s", exc)
    return stories


def _fetch_gcalendar_today() -> tuple[List[Dict[str, Any]], str]:
    """Read today's schedule via connected ``gcalendar`` CLI (strictly readonly)."""
    if not shutil.which("gcalendar"):
        return [], "not_installed"
    try:
        proc = subprocess.run(
            ["gcalendar", "readonly", "today", "--json"],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return [], "not_authenticated"
        raw = json.loads(proc.stdout)
        items = (
            raw if isinstance(raw, list) else raw.get("events", raw.get("items", []))
        )
        events: List[Dict[str, Any]] = []
        for ev in items[:12]:
            if not isinstance(ev, dict):
                continue
            start = ev.get("start", "")
            if isinstance(start, dict):
                start = start.get("dateTime") or start.get("date") or ""
            events.append(
                {
                    "summary": str(
                        ev.get("summary") or ev.get("title") or "(Untitled event)"
                    ),
                    "start": str(start),
                    "location": str(ev.get("location") or ""),
                }
            )
        return events, "gcalendar_cli"
    except Exception as exc:
        logger.debug("gcalendar readonly today failed: %s", exc)
        return [], "unavailable"


def _fetch_gdrive_recent(limit: int = 5) -> tuple[List[Dict[str, Any]], str]:
    """Read recent Google Drive files via connected ``gdrive`` CLI (strictly readonly)."""
    if not shutil.which("gdrive"):
        return [], "not_installed"
    try:
        proc = subprocess.run(
            ["gdrive", "readonly", "recent", "--max", str(limit), "--json"],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return [], "not_authenticated"
        raw = json.loads(proc.stdout)
        items = raw if isinstance(raw, list) else raw.get("files", raw.get("items", []))
        files: List[Dict[str, Any]] = []
        for f in items[:limit]:
            if not isinstance(f, dict):
                continue
            files.append(
                {
                    "name": str(f.get("name") or f.get("title") or "Untitled"),
                    "mime_type": str(f.get("mimeType") or f.get("type") or ""),
                    "modified": str(f.get("modifiedTime") or f.get("modified") or ""),
                }
            )
        return files, "gdrive_cli"
    except Exception as exc:
        logger.debug("gdrive readonly recent failed: %s", exc)
        return [], "unavailable"


class MorningBriefingService:
    """Generates executive written & spoken briefings using OpenJarvis + Workspace data."""

    def __init__(
        self,
        *,
        fact_store: Optional[LocalFactStore] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self._fact_store = fact_store or LocalFactStore()
        self._config_path = config_path or (
            get_config_dir() / "my_jarvis_briefing.json"
        )
        self._config = self._load_config()

    def _load_config(self) -> BriefingConfig:
        if not self._config_path.exists():
            return BriefingConfig()
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            return BriefingConfig(
                city=str(data.get("city", "San Francisco")),
                include_calendar=bool(data.get("include_calendar", True)),
                include_drive=bool(data.get("include_drive", True)),
                include_weather=bool(data.get("include_weather", True)),
                include_news=bool(data.get("include_news", True)),
                news_limit=int(data.get("news_limit", 5)),
                schedule_cron=str(data.get("schedule_cron", "0 8 * * *")),
            )
        except Exception:
            return BriefingConfig()

    def save_config(self, **updates: Any) -> BriefingConfig:
        for k, v in updates.items():
            if hasattr(self._config, k) and v is not None:
                setattr(self._config, k, v)
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._config.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return self._config

    @property
    def config(self) -> BriefingConfig:
        return self._config

    def generate(
        self,
        *,
        city: Optional[str] = None,
        include_network: bool = True,
    ) -> BriefingReport:
        target_city = (city or self._config.city or "San Francisco").strip()
        greeting = _time_greeting()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        weather: Dict[str, Any] = {}
        if self._config.include_weather and include_network:
            weather = _fetch_weather(target_city)

        cal_events: List[Dict[str, Any]] = []
        cal_src = "disabled"
        if self._config.include_calendar:
            cal_events, cal_src = _fetch_gcalendar_today()

        drive_files: List[Dict[str, Any]] = []
        drive_src = "disabled"
        if self._config.include_drive:
            drive_files, drive_src = _fetch_gdrive_recent(limit=5)

        news: List[Dict[str, Any]] = []
        if self._config.include_news and include_network:
            news = _fetch_hackernews(limit=self._config.news_limit)

        memories = [f.text for f in self._fact_store.list() if f.trusted_for_recall][
            -8:
        ]

        # Build clean Markdown + spoken executive script
        md_lines = [f"### {greeting}. Here is your J.A.R.V.I.S. Briefing ({now_iso})"]
        spoken_parts = [f"{greeting}. Here is your daily Jarvis briefing."]

        if weather.get("available"):
            w_line = (
                f"{weather['city']}: **{weather['temp_f']}°F**, {weather['condition']} "
                f"(feels like {weather.get('feels_like_f', '?')}°F)"
            )
            md_lines.append(f"\n**Weather**\n- {w_line}")
            spoken_parts.append(
                f"In {weather['city']}, it is currently {weather['temp_f']} degrees and {weather['condition']}."
            )

        if cal_events:
            md_lines.append(
                f"\n**Today's Schedule** ({len(cal_events)} events via {cal_src})"
            )
            for ev in cal_events[:6]:
                md_lines.append(f"- `{ev['start']}` — **{ev['summary']}**")
            spoken_parts.append(
                f"You have {len(cal_events)} events on your calendar today, starting with {cal_events[0]['summary']}."
            )
        else:
            md_lines.append(
                f"\n**Today's Schedule**\n- No upcoming events detected (`{cal_src}` — connect Google Calendar via `jarvis connect gdrive` or `gcalendar`)."
            )

        if drive_files:
            md_lines.append(
                f"\n**Recent Google Drive Activity** ({len(drive_files)} files)"
            )
            for df in drive_files[:5]:
                md_lines.append(f"- **{df['name']}**")
            spoken_parts.append(
                f"Your most recently touched Drive document is {drive_files[0]['name']}."
            )

        if news:
            md_lines.append("\n**Top Tech & AI Headlines (Hacker News)**")
            for item in news[:5]:
                md_lines.append(f"- **{item['title']}** ({item['score']} pts)")
            spoken_parts.append(f"Top tech headlines include: {news[0]['title']}.")

        if memories:
            md_lines.append("\n**Active Personal Context (Memory Vault)**")
            for m in memories[-5:]:
                md_lines.append(f"- {m}")
            spoken_parts.append(
                f"I am keeping {len(memories)} saved personal context notes active across all brains."
            )

        spoken_parts.append("All systems nominal. Ready when you are.")

        return BriefingReport(
            generated_at=now_iso,
            greeting=greeting,
            city=target_city,
            weather=weather,
            calendar_events=cal_events,
            calendar_source=cal_src,
            drive_files=drive_files,
            drive_source=drive_src,
            news=news,
            memories=memories,
            markdown="\n".join(md_lines),
            spoken_script=" ".join(spoken_parts),
        )


__all__ = ["BriefingConfig", "BriefingReport", "MorningBriefingService"]
