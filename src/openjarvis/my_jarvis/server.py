"""Extended OpenJarvis FastAPI server powering the My-JARVIS Command Center HUD.

Wraps Stanford OpenJarvis's ``create_app()`` so every native OpenJarvis endpoint
(``/v1/chat/completions``, ``/v1/models``, ``/v1/agents``, ``/health``) remains
available alongside ``/api/jarvis/*`` multi-brain, voice, memory, hardware,
and defensive self-audit routes.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from openjarvis.my_jarvis.audit import audit_local_repo, audit_website_posture
from openjarvis.my_jarvis.hardware import (
    inspect_hardware,
    recommend_models_for_hardware,
    search_huggingface_gguf,
)
from openjarvis.my_jarvis.runtime import MyJarvisRuntime
from openjarvis.server.app import create_app

logger = logging.getLogger(__name__)

STATIC_DIR = pathlib.Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=16000)
    brain_id: Optional[str] = None
    use_tools: bool = True


class BrainSwitchRequest(BaseModel):
    target: str = Field(..., min_length=1, max_length=200)


class CustomBrainRequest(BaseModel):
    brain_id: str = Field(..., min_length=1, max_length=80)
    name: str = Field(..., min_length=1, max_length=120)
    provider: str = Field(..., min_length=1, max_length=40)
    model: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    host: str = ""
    env_var: str = ""


class ProviderConfigRequest(BaseModel):
    provider: str = Field(..., min_length=1, max_length=40)
    api_key: str = ""
    host: str = ""


class MemoryAddRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source: str = "hud"


class SiteAuditRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)


class RepoAuditRequest(BaseModel):
    path: str = Field(".", min_length=1, max_length=2000)


class PersonaUpdateRequest(BaseModel):
    persona: str = Field(..., min_length=10, max_length=4000)


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------


def _sync_openjarvis_state(app: FastAPI, runtime: MyJarvisRuntime) -> None:
    """Keep OpenJarvis's native /v1/* state synced with the hot-swapped brain."""
    active = runtime.router.active_brain
    app.state.engine = runtime.router.get_engine_for(active)
    app.state.model = active.model
    app.state.engine_name = active.provider


def create_my_jarvis_app(
    runtime: Optional[MyJarvisRuntime] = None,
    *,
    cors_origins: Optional[List[str]] = None,
) -> FastAPI:
    """Create the combined OpenJarvis + My-JARVIS FastAPI application."""
    rt = runtime or MyJarvisRuntime()
    active = rt.router.active_brain
    initial_engine = rt.router.get_engine_for(active)

    origins = cors_origins or [
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    app = create_app(
        engine=initial_engine,
        model=active.model,
        bus=rt.bus,
        engine_name=active.provider,
        cors_origins=origins,
    )
    app.title = "My-JARVIS (Built on Stanford OpenJarvis)"
    app.state.my_jarvis = rt

    api = APIRouter(prefix="/api/jarvis", tags=["my-jarvis"])

    @api.get("/status")
    async def get_status() -> Dict[str, Any]:
        import openjarvis

        return {
            "ok": True,
            "openjarvis_version": openjarvis.__version__,
            "active_brain": rt.router.active_brain.to_dict(),
            "active_brain_ready": rt.router.is_brain_configured(rt.router.active_brain),
            "memory_count": len(rt.list_memories()),
            "history_count": len(rt.get_history()),
            "persona": rt.persona,
        }

    @api.post("/persona")
    async def update_persona(req: PersonaUpdateRequest) -> Dict[str, Any]:
        rt.persona = req.persona.strip()
        return {"ok": True, "persona": rt.persona}

    @api.get("/history")
    async def get_history() -> Dict[str, Any]:
        return {"messages": rt.get_history()}

    @api.delete("/history")
    async def clear_history() -> Dict[str, Any]:
        cleared = rt.clear_history()
        return {"ok": True, "cleared": cleared}

    @api.post("/chat")
    async def post_chat(req: ChatRequest) -> Dict[str, Any]:
        try:
            result = rt.chat(
                req.message,
                brain_id=req.brain_id,
                use_tools=req.use_tools,
            )
            _sync_openjarvis_state(app, rt)
            return result
        except Exception as exc:
            logger.exception("Unhandled error in /api/jarvis/chat")
            return JSONResponse(
                status_code=200,
                content={
                    "reply": {
                        "id": "err",
                        "role": "assistant",
                        "content": f"⚠️ Request encountered an unexpected error: `{exc}`",
                        "brain_id": rt.router.active_brain.id,
                        "model": rt.router.active_brain.model,
                        "provider": rt.router.active_brain.provider,
                        "latency_ms": 0.0,
                        "tools_used": [],
                    },
                    "active_brain": rt.router.active_brain.to_dict(),
                    "brain_switched": False,
                },
            )

    @api.get("/brains")
    async def list_brains() -> Dict[str, Any]:
        return {
            "active_brain_id": rt.router.active_brain.id,
            "brains": rt.router.list_brains(probe_live=True),
            "providers": rt.router.get_provider_status(),
        }

    @api.post("/brains/switch")
    async def switch_brain(req: BrainSwitchRequest) -> Dict[str, Any]:
        res = rt.router.switch_brain(req.target)
        if not res.get("ok"):
            raise HTTPException(
                status_code=400, detail=res.get("error", "Invalid brain")
            )
        _sync_openjarvis_state(app, rt)
        return res

    @api.post("/brains/custom")
    async def add_custom_brain(req: CustomBrainRequest) -> Dict[str, Any]:
        try:
            bp = rt.router.add_custom_brain(
                brain_id=req.brain_id,
                name=req.name,
                provider=req.provider,
                model=req.model,
                description=req.description,
                host=req.host,
                env_var=req.env_var,
            )
            return {"ok": True, "brain": bp.to_dict()}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @api.post("/providers")
    async def configure_provider(req: ProviderConfigRequest) -> Dict[str, Any]:
        try:
            res = rt.router.configure_provider(
                provider=req.provider,
                api_key=req.api_key,
                host=req.host,
            )
            _sync_openjarvis_state(app, rt)
            return res
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @api.get("/memory")
    async def get_memory() -> Dict[str, Any]:
        mems = rt.list_memories()
        return {"count": len(mems), "facts": mems}

    @api.post("/memory")
    async def add_memory(req: MemoryAddRequest) -> Dict[str, Any]:
        res = rt.add_memory(req.text, source=req.source)
        if not res.get("ok"):
            raise HTTPException(
                status_code=400, detail=res.get("error", "Failed to save memory")
            )
        return res

    @api.delete("/memory/{index}")
    async def delete_memory(index: int) -> Dict[str, Any]:
        res = rt.delete_memory(index)
        if not res.get("ok"):
            raise HTTPException(
                status_code=404, detail=res.get("error", "Memory not found")
            )
        return res

    @api.get("/hardware")
    async def get_hardware() -> Dict[str, Any]:
        profile = inspect_hardware(rt.router.ollama_host)
        recs = recommend_models_for_hardware(profile)
        return {
            "hardware": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
        }

    @api.get("/hf-models")
    async def get_hf_models(
        q: str = Query(default="abliterated", min_length=1, max_length=100),
        limit: int = Query(default=8, ge=1, le=20),
    ) -> Dict[str, Any]:
        return search_huggingface_gguf(q, limit=limit)

    @api.post("/audit/site")
    async def run_site_audit(req: SiteAuditRequest) -> Dict[str, Any]:
        report = audit_website_posture(req.url)
        return report.to_dict()

    @api.post("/audit/repo")
    async def run_repo_audit(req: RepoAuditRequest) -> Dict[str, Any]:
        return audit_local_repo(req.path)

    @api.get("/doctor")
    async def run_doctor() -> Dict[str, Any]:
        return rt.run_doctor()

    app.include_router(api)

    # Mount the custom JARVIS HUD UI cleanly
    if STATIC_DIR.is_dir():
        app.mount(
            "/jarvis-static",
            StaticFiles(directory=STATIC_DIR),
            name="my-jarvis-static",
        )

        @app.get("/", include_in_schema=False)
        async def serve_hud_root() -> FileResponse:
            return FileResponse(
                STATIC_DIR / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

    return app


__all__ = ["create_my_jarvis_app"]
