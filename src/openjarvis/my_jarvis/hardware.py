"""Hardware profiling and Hugging Face / Ollama local model fit advisor.

Inspects the local machine's CPU, RAM, disk, and GPU/VRAM (NVIDIA or Apple
Silicon unified memory) and recommends open-weight models (including GGUF
quants on Hugging Face and Ollama) that fit comfortably on the user's hardware.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import httpx
import psutil

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    name: str
    vram_gb: float
    vendor: str
    unified_memory: bool = False


@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    accelerator: str = "cpu"
    ollama_installed: bool = False
    ollama_running: bool = False
    ollama_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendedModel:
    id: str
    name: str
    params: str
    quant: str
    min_ram_gb: float
    recommended_vram_gb: float
    category: str
    fit: str  # "optimal" | "tight" | "cloud_recommended"
    fit_reason: str
    ollama_cmd: str
    hf_repo: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Curated open-weight & abliterated/uncensored local models inspired by
# Stanford OpenJarvis + community GGUF deployments.
CURATED_LOCAL_MODELS: List[Dict[str, Any]] = [
    {
        "id": "qwen3:4b",
        "name": "Qwen 3 4B",
        "params": "4B",
        "quant": "Q4_K_M",
        "min_ram_gb": 4.0,
        "recommended_vram_gb": 4.0,
        "category": "fast-daily",
        "ollama_cmd": "ollama pull qwen3:4b",
        "hf_repo": "Qwen/Qwen3-4B-GGUF",
        "notes": "Ultra-fast local assistant; runs smoothly even on 8 GB CPU-only laptops.",
    },
    {
        "id": "qwen3:8b",
        "name": "Qwen 3 8B",
        "params": "8B",
        "quant": "Q4_K_M",
        "min_ram_gb": 8.0,
        "recommended_vram_gb": 6.5,
        "category": "balanced",
        "ollama_cmd": "ollama pull qwen3:8b",
        "hf_repo": "Qwen/Qwen3-8B-GGUF",
        "notes": "OpenJarvis sweet spot: strong tool calling, reasoning, and fast local speed.",
    },
    {
        "id": "deepseek-r1:8b",
        "name": "DeepSeek-R1 Distill 8B",
        "params": "8B",
        "quant": "Q4_K_M",
        "min_ram_gb": 8.0,
        "recommended_vram_gb": 6.5,
        "category": "reasoning",
        "ollama_cmd": "ollama pull deepseek-r1:8b",
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "notes": "Step-by-step reasoning locally with zero cloud calls.",
    },
    {
        "id": "dolphin-llama3:8b",
        "name": "Dolphin 3 / Llama 3 8B (Uncensored)",
        "params": "8B",
        "quant": "Q4_K_M",
        "min_ram_gb": 8.0,
        "recommended_vram_gb": 6.5,
        "category": "uncensored",
        "ollama_cmd": "ollama pull dolphin-llama3:8b",
        "hf_repo": "bartowski/Dolphin3.0-Llama3.1-8B-GGUF",
        "notes": "Popular open-weight uncensored model for direct local answers on 8-16 GB RAM.",
    },
    {
        "id": "qwen2.5-coder:14b",
        "name": "Qwen 2.5 Coder 14B",
        "params": "14B",
        "quant": "Q4_K_M",
        "min_ram_gb": 16.0,
        "recommended_vram_gb": 11.0,
        "category": "coding",
        "ollama_cmd": "ollama pull qwen2.5-coder:14b",
        "hf_repo": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "notes": "High-accuracy local coding and refactoring agent for 16 GB+ machines.",
    },
    {
        "id": "qwen3:30b-a3b",
        "name": "Qwen 3 30B-A3B (MoE)",
        "params": "30B (3B active)",
        "quant": "Q4_K_M",
        "min_ram_gb": 24.0,
        "recommended_vram_gb": 20.0,
        "category": "moe-flagship",
        "ollama_cmd": "ollama pull qwen3:30b-a3b",
        "hf_repo": "Qwen/Qwen3-30B-A3B-GGUF",
        "notes": "Mixture-of-Experts speed with large-model knowledge on 24-32 GB Apple Silicon/GPU.",
    },
    {
        "id": "qwen3:32b",
        "name": "Qwen 3 27B/32B Dense (Abliterated-Ready)",
        "params": "27B–32B",
        "quant": "Q4_K_M",
        "min_ram_gb": 24.0,
        "recommended_vram_gb": 22.0,
        "category": "heavy-local",
        "ollama_cmd": "ollama pull qwen3:32b",
        "hf_repo": "huihui-ai/Qwen3-32B-abliterated",
        "notes": "Matches the 27B-32B tier highlighted for local workstations (24 GB VRAM / 32 GB Mac).",
    },
    {
        "id": "deepseek-r1:70b",
        "name": "DeepSeek-R1 70B",
        "params": "70B",
        "quant": "Q4_K_M",
        "min_ram_gb": 48.0,
        "recommended_vram_gb": 44.0,
        "category": "workstation",
        "ollama_cmd": "ollama pull deepseek-r1:70b",
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "notes": "Flagship local reasoning for 64 GB+ Apple Silicon or dual-GPU rigs; otherwise use Venice/Cloud.",
    },
]


def _detect_gpus(ram_total_gb: float) -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    # 1. Try NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        name = parts[0]
                        vram_mb = float(parts[1])
                        gpus.append(
                            GPUInfo(
                                name=name,
                                vram_gb=round(vram_mb / 1024.0, 2),
                                vendor="nvidia",
                                unified_memory=False,
                            )
                        )
        except Exception as exc:
            logger.debug("nvidia-smi probe failed: %s", exc)

    # 2. Apple Silicon unified memory
    if not gpus and platform.system() == "Darwin" and platform.machine() == "arm64":
        # macOS allows ~75% of unified RAM to be wired for Metal inference.
        gpus.append(
            GPUInfo(
                name="Apple Silicon GPU (Metal Unified Memory)",
                vram_gb=round(ram_total_gb * 0.75, 2),
                vendor="apple",
                unified_memory=True,
            )
        )
    return gpus


def _probe_ollama(host: str | None = None) -> tuple[bool, bool, List[str]]:
    installed = shutil.which("ollama") is not None
    base = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip(
        "/"
    )
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [
                str(m.get("name", ""))
                for m in data.get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]
            return installed, True, models
    except Exception:
        pass
    return installed, False, []


def inspect_hardware(ollama_host: str | None = None) -> HardwareProfile:
    """Collect a real-time hardware profile of the current machine."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.expanduser("~"))
    ram_total_gb = round(vm.total / (1024**3), 2)
    ram_avail_gb = round(vm.available / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    gpus = _detect_gpus(ram_total_gb)
    total_vram = round(sum(g.vram_gb for g in gpus), 2)
    if any(g.vendor == "nvidia" for g in gpus):
        accel = "nvidia_cuda"
    elif any(g.vendor == "apple" for g in gpus):
        accel = "apple_metal"
    else:
        accel = "cpu"

    ollama_installed, ollama_running, ollama_models = _probe_ollama(ollama_host)

    return HardwareProfile(
        os_name=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_avail_gb,
        disk_free_gb=disk_free_gb,
        gpus=gpus,
        total_vram_gb=total_vram,
        accelerator=accel,
        ollama_installed=ollama_installed,
        ollama_running=ollama_running,
        ollama_models=ollama_models,
    )


def recommend_models_for_hardware(
    profile: HardwareProfile,
) -> List[RecommendedModel]:
    """Score curated local models against actual RAM/VRAM headroom."""
    effective_gpu = profile.total_vram_gb
    effective_ram = profile.ram_total_gb

    recommendations: List[RecommendedModel] = []
    for entry in CURATED_LOCAL_MODELS:
        req_vram = float(entry["recommended_vram_gb"])
        req_ram = float(entry["min_ram_gb"])
        if effective_gpu >= req_vram:
            fit = "optimal"
            reason = f"Fits in GPU VRAM ({effective_gpu:.1f} GB available vs {req_vram:.1f} GB needed)."
        elif effective_ram >= req_ram * 1.25:
            fit = "optimal" if effective_gpu == 0 and req_ram <= 4.5 else "tight"
            reason = (
                f"Runs on system RAM ({effective_ram:.1f} GB RAM; CPU/offload speed)."
                if effective_gpu == 0
                else f"Partial GPU offload ({effective_gpu:.1f} GB VRAM + {effective_ram:.1f} GB RAM)."
            )
        elif effective_ram >= req_ram:
            fit = "tight"
            reason = f"Close to RAM ceiling ({effective_ram:.1f} GB total vs {req_ram:.1f} GB minimum)."
        else:
            fit = "cloud_recommended"
            reason = (
                f"Needs ~{req_ram:.0f} GB RAM / {req_vram:.0f} GB VRAM — "
                "use Venice AI or Cloud brain on this machine."
            )

        recommendations.append(
            RecommendedModel(
                id=entry["id"],
                name=entry["name"],
                params=entry["params"],
                quant=entry["quant"],
                min_ram_gb=req_ram,
                recommended_vram_gb=req_vram,
                category=entry["category"],
                fit=fit,
                fit_reason=reason,
                ollama_cmd=entry["ollama_cmd"],
                hf_repo=entry["hf_repo"],
                notes=entry["notes"],
            )
        )
    return recommendations


def search_huggingface_gguf(
    query: str = "abliterated",
    *,
    limit: int = 10,
) -> Dict[str, Any]:
    """Query Hugging Face Hub for GGUF models runnable directly via Ollama.

    Returns live Hub results when outbound internet access is available, or
    cleanly falls back to filtering the curated local model catalog.
    """
    q = (query or "qwen").strip()
    url = "https://huggingface.co/api/models"
    params = {
        "search": q,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": max(1, min(limit, 25)),
    }
    try:
        resp = httpx.get(url, params=params, timeout=5.0)
        if resp.status_code == 200:
            items = resp.json()
            results: List[Dict[str, Any]] = []
            for item in items:
                repo_id = item.get("id") or item.get("modelId")
                if not repo_id:
                    continue
                results.append(
                    {
                        "repo_id": repo_id,
                        "downloads": int(item.get("downloads", 0) or 0),
                        "likes": int(item.get("likes", 0) or 0),
                        "tags": (item.get("tags") or [])[:6],
                        "ollama_run_cmd": f"ollama run hf.co/{repo_id}",
                        "hf_url": f"https://huggingface.co/{repo_id}",
                    }
                )
            if results:
                return {"source": "huggingface_hub", "query": q, "models": results}
    except Exception as exc:
        logger.debug(
            "Hugging Face Hub search unavailable (%s); using curated fallback", exc
        )

    # Offline / blocked-network curated fallback
    lower = q.lower()
    matched = [
        m
        for m in CURATED_LOCAL_MODELS
        if lower in m["name"].lower()
        or lower in m["hf_repo"].lower()
        or lower in m["category"].lower()
        or lower in m["notes"].lower()
    ] or CURATED_LOCAL_MODELS
    return {
        "source": "curated_offline_catalog",
        "query": q,
        "models": [
            {
                "repo_id": m["hf_repo"],
                "downloads": 0,
                "likes": 0,
                "tags": ["gguf", m["category"], m["params"]],
                "ollama_run_cmd": (
                    f"ollama run hf.co/{m['hf_repo']}"
                    if "gguf" in m["hf_repo"].lower()
                    else m["ollama_cmd"]
                ),
                "hf_url": f"https://huggingface.co/{m['hf_repo']}",
                "notes": m["notes"],
            }
            for m in matched[:limit]
        ],
    }


__all__ = [
    "CURATED_LOCAL_MODELS",
    "GPUInfo",
    "HardwareProfile",
    "RecommendedModel",
    "inspect_hardware",
    "recommend_models_for_hardware",
    "search_huggingface_gguf",
]
