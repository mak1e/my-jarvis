"""Automated tests for My-JARVIS extensions built on top of Stanford OpenJarvis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi.testclient import TestClient

from openjarvis.core.registry import EngineRegistry, ToolRegistry
from openjarvis.core.types import Message, Role
from openjarvis.engine._base import InferenceEngine
from openjarvis.memory.store import LocalFactStore
from openjarvis.my_jarvis import register_my_jarvis_primitives
from openjarvis.my_jarvis.audit import audit_local_repo, audit_website_posture
from openjarvis.my_jarvis.brains import (
    BrainRouter,
    detect_brain_switch_intent,
    mask_secret,
)
from openjarvis.my_jarvis.hardware import (
    GPUInfo,
    HardwareProfile,
    recommend_models_for_hardware,
)
from openjarvis.my_jarvis.runtime import MyJarvisRuntime
from openjarvis.my_jarvis.server import create_my_jarvis_app
from openjarvis.my_jarvis.venice_engine import VeniceEngine


class _FakeLLMEngine(InferenceEngine):
    """Minimal mock OpenJarvis engine to verify live OrchestratorAgent path."""

    engine_id = "fake_test"
    is_cloud = True

    def generate(self, messages, *, model: str, **kwargs: Any) -> Dict[str, Any]:
        sys_msg = next((m.text for m in messages if m.role == Role.SYSTEM), "")
        user_msg = next((m.text for m in reversed(messages) if m.role == Role.USER), "")
        return {
            "content": (
                f"[mock:{model}] Heard '{user_msg}'. HasMemory={'32GB Mac' in sys_msg}"
            ),
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            "model": model,
            "finish_reason": "stop",
        }

    async def stream(self, messages, *, model: str, **kwargs: Any):
        yield self.generate(messages, model=model)["content"]

    def list_models(self) -> List[str]:
        return ["fake-model"]

    def health(self) -> bool:
        return True


def test_venice_engine_registration_and_generate() -> None:
    register_my_jarvis_primitives()

    assert EngineRegistry.contains("venice")
    captured: Dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-r1-671b",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Direct answer."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    eng = VeniceEngine(api_key="venice-test-key-12345")
    eng._client = httpx.Client(
        base_url=eng._host,
        headers=eng._headers,
        transport=httpx.MockTransport(_handler),
    )
    res = eng.generate(
        [Message(role=Role.USER, content="Hello")], model="venice/deepseek-r1-671b"
    )
    assert res["content"] == "Direct answer."
    assert captured["url"] == "https://api.venice.ai/api/v1/chat/completions"
    assert captured["auth"] == "Bearer venice-test-key-12345"
    assert captured["body"]["model"] == "deepseek-r1-671b"
    assert captured["body"]["venice_parameters"] == {
        "include_venice_system_prompt": False
    }


def test_voice_and_slash_brain_switch_detection() -> None:
    assert detect_brain_switch_intent("/brain venice-deepseek") == "venice-deepseek"
    assert (
        detect_brain_switch_intent("Hey Jarvis, switch your brain to DeepSeek!")
        == "DeepSeek"
    )
    assert detect_brain_switch_intent("Switch to Claude Opus") == "Claude Opus"
    assert detect_brain_switch_intent("Use the local qwen model.") == "local qwen"
    assert detect_brain_switch_intent("What is the capital of France?") is None


def test_brain_router_fuzzy_resolution_and_custom_brain(tmp_path: Path) -> None:
    router = BrainRouter(state_path=tmp_path / "brains.json")
    assert router.active_brain.id == "builtin-offline"
    assert router.resolve_brain("deepseek").id == "venice-deepseek"
    assert router.resolve_brain("opus").id == "claude-opus"
    assert router.resolve_brain("local qwen").id == "local-qwen"

    bp = router.add_custom_brain(
        brain_id="hf-qwen-abliterated",
        name="Qwen 3 Abliterated",
        provider="ollama",
        model="hf.co/huihui-ai/Qwen3-8B-abliterated-GGUF",
    )
    assert bp.id == "hf-qwen-abliterated"
    res = router.switch_brain("hf-qwen-abliterated")
    assert res["ok"] is True
    assert router.active_brain.id == "hf-qwen-abliterated"
    assert mask_secret("sk-1234567890abcdef") == "sk-1••••cdef"


def test_hardware_recommendations_scale_with_vram() -> None:
    low_cpu = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores_physical=2,
        cpu_cores_logical=4,
        ram_total_gb=8.0,
        ram_available_gb=6.5,
        disk_free_gb=50.0,
    )
    low_recs = {r.id: r for r in recommend_models_for_hardware(low_cpu)}
    assert low_recs["qwen3:4b"].fit == "optimal"
    assert low_recs["deepseek-r1:70b"].fit == "cloud_recommended"

    apple_64 = HardwareProfile(
        os_name="Darwin",
        arch="arm64",
        cpu_cores_physical=14,
        cpu_cores_logical=14,
        ram_total_gb=64.0,
        ram_available_gb=54.0,
        disk_free_gb=400.0,
        gpus=[
            GPUInfo(
                name="Apple M4 Max", vram_gb=48.0, vendor="apple", unified_memory=True
            )
        ],
        total_vram_gb=48.0,
        accelerator="apple_metal",
    )
    high_recs = {r.id: r for r in recommend_models_for_hardware(apple_64)}
    assert high_recs["qwen3:32b"].fit == "optimal"
    assert high_recs["deepseek-r1:70b"].fit == "optimal"


def test_defensive_audit_ssrf_block_and_repo_scanner(tmp_path: Path) -> None:
    # Cloud metadata endpoint must always be refused
    blocked = audit_website_posture("http://169.254.169.254/latest/meta-data/")
    assert blocked.error != ""
    assert "blocked" in blocked.error.lower()

    # Local repo hygiene scan detects masked credential & shell=True
    sample_py = tmp_path / "app.py"
    sample_py.write_text(
        'import subprocess\nTOKEN = "ghp_1234567890abcdefghij123456"\nsubprocess.run("ls", shell=True)\n',
        encoding="utf-8",
    )
    rep = audit_local_repo(str(tmp_path))
    assert rep["risk_level"] == "HIGH"
    titles = [f["title"] for f in rep["findings"]]
    assert any("GitHub Personal Access Token" in t for t in titles)
    assert any("shell=True" in t for t in titles)
    # Verify raw token is masked in findings
    assert "1234567890abcdefghij" not in json.dumps(rep)


def test_custom_tools_registered_in_openjarvis_registry() -> None:
    register_my_jarvis_primitives()

    for tool_name in (
        "hardware_advisor",
        "hf_model_search",
        "site_security_audit",
        "repo_security_audit",
        "jarvis_memory",
        "switch_brain",
    ):
        assert ToolRegistry.contains(tool_name), f"Tool {tool_name} not registered"


def test_runtime_end_to_end_memory_brain_switch_and_mock_llm(tmp_path: Path) -> None:
    router = BrainRouter(state_path=tmp_path / "brains.json")
    store = LocalFactStore(path=tmp_path / "facts.jsonl")
    rt = MyJarvisRuntime(
        router=router,
        fact_store=store,
        session_path=tmp_path / "history.json",
    )

    # 1. Remember fact via natural language
    r1 = rt.chat("Remember that I develop on a 32GB Mac")
    assert "32GB Mac" in r1["reply"]["content"]
    assert len(rt.list_memories()) == 1

    # 2. Run offline AST calculator tool
    r2 = rt.chat("calculate (137 * 42) + 10")
    assert "5764" in r2["reply"]["content"]

    # 3. Switch brain via voice-style command
    r3 = rt.chat("Hey Jarvis, switch your brain to Venice DeepSeek")
    assert r3["brain_switched"] is True
    assert rt.router.active_brain.id == "venice-deepseek"

    # 4. Inject mock engine to prove memory flows across brain switch through OrchestratorAgent
    router._engine_cache["venice-deepseek"] = _FakeLLMEngine()
    monkey_orig = router.is_brain_configured
    router.is_brain_configured = lambda _b: True  # type: ignore[method-assign]
    try:
        r4 = rt.chat("Confirm my dev setup", use_tools=True)
        assert "HasMemory=True" in r4["reply"]["content"]
    finally:
        router.is_brain_configured = monkey_orig  # type: ignore[method-assign]


def test_fastapi_server_routes_and_openjarvis_compat(tmp_path: Path) -> None:
    router = BrainRouter(state_path=tmp_path / "brains.json")
    store = LocalFactStore(path=tmp_path / "facts.jsonl")
    rt = MyJarvisRuntime(
        router=router,
        fact_store=store,
        session_path=tmp_path / "history.json",
    )
    app = create_my_jarvis_app(rt)
    client = TestClient(app)

    # HUD HTML & JS served cleanly
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "J.A.R.V.I.S." in root_resp.text

    js_resp = client.get("/jarvis-static/app.js")
    assert js_resp.status_code == 200

    # Status + Brains + Hardware + Doctor
    st = client.get("/api/jarvis/status")
    assert st.status_code == 200
    assert st.json()["ok"] is True

    brains = client.get("/api/jarvis/brains")
    assert brains.status_code == 200
    assert len(brains.json()["brains"]) >= 10

    hw = client.get("/api/jarvis/hardware")
    assert hw.status_code == 200
    assert hw.json()["hardware"]["ram_total_gb"] > 0

    doc = client.get("/api/jarvis/doctor")
    assert doc.status_code == 200
    assert len(doc.json()["checks"]) >= 5

    # Memory CRUD
    m_add = client.post("/api/jarvis/memory", json={"text": "Prefer concise answers"})
    assert m_add.status_code == 200
    m_list = client.get("/api/jarvis/memory")
    assert m_list.json()["count"] == 1
    m_del = client.delete("/api/jarvis/memory/0")
    assert m_del.status_code == 200
    assert client.get("/api/jarvis/memory").json()["count"] == 0

    # Chat + Voice Brain Switch via API
    chat_resp = client.post(
        "/api/jarvis/chat", json={"message": "Switch brain to Claude Opus"}
    )
    assert chat_resp.status_code == 200
    assert chat_resp.json()["brain_switched"] is True
    assert chat_resp.json()["active_brain"]["id"] == "claude-opus"

    # Native Stanford OpenJarvis OpenAI-compatible route still works!
    models_resp = client.get("/v1/models")
    assert models_resp.status_code == 200
