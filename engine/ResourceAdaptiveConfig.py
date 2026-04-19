"""
ResourceAdaptiveConfig — Hardware-aware auto-scaling for HaromaX6.

Probes the runtime environment (CPU, RAM, GPU, disk, network, sensors)
and produces a tiered configuration that every module consults at init.

Tiers
-----
  0  embedded   : Raspberry Pi, Jetson Nano, <4 GB RAM, no GPU.
                  Hash-bucket encoder, template composer, no LLM.
  1  edge       : Desktop / laptop, 4-16 GB, small or no GPU.
                  MiniLM embeddings, optional 1-3B local GGUF.
  2  workstation: 16-64 GB, decent GPU (8-24 GB VRAM).
                  Full pretrained encoders, 7-13B local GGUF.
  3  server     : 64+ GB, multi-GPU or high-VRAM card.
                  Large local models (30-70B), all modules max.
  4  cloud      : Datacenter / managed infra, or API keys detected.
                  Full local stack PLUS frontier API (GPT-5, Claude,
                  Gemini) for generation, reasoning, imagination.

The config is a singleton — modules call ``get_resource_config()``
and read their section.  Every setting has a sane default so the
system always boots, even on unknown hardware.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import platform
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


# ======================================================================
# Hardware probing
# ======================================================================


def _get_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024**2)
        except Exception as _e:
            print(f"[ResourceAdaptiveConfig] /proc/meminfo error: {_e}", flush=True)
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                "wmic OS get TotalVisibleMemorySize /Value", shell=True, text=True, timeout=5
            )
            for line in out.strip().splitlines():
                if "=" in line:
                    return int(line.split("=")[1].strip()) / (1024**2)
        except Exception as _e:
            print(f"[ResourceAdaptiveConfig] wmic error: {_e}", flush=True)
    return 4.0


def _get_cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _get_gpu_info() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "name": "",
        "vram_gb": 0.0,
        "count": 0,
        "cuda": False,
        "mps": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            result["cuda"] = True
            result["available"] = True
            result["count"] = torch.cuda.device_count()
            result["name"] = torch.cuda.get_device_name(0)
            result["vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["mps"] = True
            result["available"] = True
            result["name"] = "Apple Silicon (MPS)"
    except (ImportError, OSError):
        # OSError: e.g. WinError 1114 when torch's native DLLs fail to load
        pass
    return result


def _is_raspberry_pi() -> bool:
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/cpuinfo") as f:
            return "raspberry" in f.read().lower()
    except Exception:
        return False
    return False


def _detect_api_keys() -> Dict[str, bool]:
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
        "ollama": _ollama_running(),
    }


def _ollama_running() -> bool:
    try:
        import urllib.request

        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            _ = resp.read()
            return resp.status == 200
    except Exception:
        return False


def _find_gguf_models(models_dir: str) -> List[Dict[str, Any]]:
    found = []
    if not os.path.isdir(models_dir):
        return found
    for fname in sorted(os.listdir(models_dir)):
        if fname.lower().endswith(".gguf"):
            path = os.path.join(models_dir, fname)
            size_gb = os.path.getsize(path) / (1024**3)
            found.append({"name": fname, "path": path, "size_gb": round(size_gb, 1)})
    return found


def _is_multi_shard_gguf(filename: str) -> bool:
    """True if filename looks like a split model (e.g. ...-00001-of-00002.gguf)."""
    return bool(re.search(r"\d+-of-\d+", filename.lower()))


def _pick_auto_gguf(models: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose a single-file quantized GGUF from scans (prefer Q4_K_M-style over fp16 shards).

    **Gemma 4** filenames (``gemma-4``, ``gemma4``, ``gemma_4``) rank above other
    families when multiple models are present, so local defaults track the project
    soul config unless overridden.

    Among equally ranked quantizations, prefers the **smaller** file on disk so
    lighter models (e.g. 1.5B over 3B) win when both are in ``models/``.
    Set env ``HAROMA_AUTO_GGUF_PREFER=largest`` to prefer the larger file on ties.
    """
    if not models:
        return None
    non_shard = [m for m in models if not _is_multi_shard_gguf(m["name"])]
    pool = non_shard if non_shard else models
    prefer = str(
        os.environ.get("HAROMA_AUTO_GGUF_PREFER", "smallest") or "smallest",
    ).lower()

    def rank_key(m: Dict[str, Any]):
        name = m["name"].lower()
        score = 0
        # Prefer Gemma 4 family when several GGUFs are present (soul / models/).
        if "gemma-4" in name or "gemma4" in name or "gemma_4" in name:
            score += 900
        if "fp16" in name and not any(
            x in name for x in ("q4", "q5", "q6", "q8", "q3", "iq", "iq4")
        ):
            score -= 80
        prefs = (
            "q4_k_m",
            "q5_k_m",
            "q4_0",
            "iq4_xs",
            "iq4_nl",
            "iq4",
            "q3_k_m",
            "q3_k_s",
            "q5_0",
            "q6_k",
            "q8_0",
            "f16",
            "fp16",
        )
        for i, tag in enumerate(prefs):
            if tag in name:
                score += 500 - i * 12
                break
        sz = float(m.get("size_gb", 0.0))
        tie = sz if prefer == "largest" else -sz
        return (score, tie)

    return max(pool, key=rank_key)


# ======================================================================
# Tier determination
# ======================================================================


def _compute_tier(hw: Dict[str, Any]) -> int:
    ram = hw["ram_gb"]
    gpu = hw["gpu"]
    apis = hw["api_keys"]
    has_api = any(apis.values())

    if ram < 4 or hw["is_rpi"]:
        return 0
    if ram < 16 and not gpu["available"]:
        return 1
    if ram < 64 and gpu.get("vram_gb", 0) < 20:
        if has_api:
            return 4
        return 2
    if has_api:
        return 4
    return 3


TIER_NAMES = {
    0: "embedded",
    1: "edge",
    2: "workstation",
    3: "server",
    4: "cloud",
}


# ======================================================================
# Per-module configuration profiles
# ======================================================================


def _encoder_config(tier: int) -> Dict[str, Any]:
    if tier == 0:
        return {
            "mode": "hash_bucket",
            "embed_dim": 128,
            "vocab_size": 4096,
            "pretrained_model": None,
        }
    if tier == 1:
        return {
            "mode": "pretrained",
            "embed_dim": 256,
            "vocab_size": 8192,
            "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    return {
        "mode": "pretrained",
        "embed_dim": 384,
        "vocab_size": 8192,
        "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def _llm_config(tier: int, hw: Dict[str, Any]) -> Dict[str, Any]:
    apis = hw["api_keys"]
    gguf_models = hw.get("gguf_models", [])
    gpu = hw["gpu"]

    cfg: Dict[str, Any] = {
        "local_gguf": None,
        "n_ctx": 2048,
        "n_gpu_layers": 0,
        "api_provider": None,
        "api_model": None,
        "api_temperature": 0.7,
        "api_max_tokens": 512,
        "use_best_of_n": False,
        "mode": "auto",
    }

    if gguf_models:
        best = _pick_auto_gguf(gguf_models)
        if best:
            cfg["local_gguf"] = best["path"]
            print(
                f"[ResourceConfig] LLM auto-selected local: {best['name']} ({best['size_gb']} GB)",
                flush=True,
            )

    if tier == 0:
        cfg["n_ctx"] = 512
        cfg["local_gguf"] = None
    elif tier == 1:
        cfg["n_ctx"] = 4096
        if gpu["available"]:
            cfg["n_gpu_layers"] = 8
    elif tier == 2:
        cfg["n_ctx"] = 8192 if gpu["available"] else 4096
        if gpu["available"]:
            cfg["n_gpu_layers"] = -1
        cfg["use_best_of_n"] = True
    elif tier >= 3:
        cfg["n_ctx"] = 32768 if gpu["available"] else 8192
        if gpu["available"]:
            cfg["n_gpu_layers"] = -1
        cfg["use_best_of_n"] = True

    if tier == 4:
        if apis.get("openai"):
            cfg["api_provider"] = "openai"
            cfg["api_model"] = "gpt-4.1"
            cfg["api_max_tokens"] = 2048
        elif apis.get("anthropic"):
            cfg["api_provider"] = "anthropic"
            cfg["api_model"] = "claude-sonnet-4-20250514"
            cfg["api_max_tokens"] = 2048
        elif apis.get("google"):
            cfg["api_provider"] = "google"
            cfg["api_model"] = "gemini-2.5-pro"
            cfg["api_max_tokens"] = 2048
        elif apis.get("ollama"):
            cfg["api_provider"] = "ollama"
            cfg["api_model"] = "llama3:latest"
            cfg["api_max_tokens"] = 2048

    # Explicit GGUF path (Option A: local llama-cpp) overrides scan and tier-0 clear.
    _explicit_gguf = (
        os.environ.get("HAROMA_LOCAL_GGUF") or os.environ.get("ELARION_LOCAL_GGUF") or ""
    ).strip()
    if _explicit_gguf and os.path.isfile(_explicit_gguf):
        cfg["local_gguf"] = os.path.abspath(_explicit_gguf)
        print(
            f"[ResourceConfig] LLM local GGUF (env): {cfg['local_gguf']}",
            flush=True,
        )

    return cfg


def _memory_config(tier: int) -> Dict[str, Any]:
    if tier == 0:
        return {
            "max_nodes": 5000,
            "faiss_enabled": False,
            "recall_limit": 5,
            "index_rebuild_interval": 50,
        }
    if tier == 1:
        return {
            "max_nodes": 50000,
            "faiss_enabled": True,
            "recall_limit": 10,
            "index_rebuild_interval": 20,
        }
    return {
        "max_nodes": 500000,
        "faiss_enabled": True,
        "recall_limit": 20,
        "index_rebuild_interval": 10,
    }


def _cycle_config(tier: int) -> Dict[str, Any]:
    if tier == 0:
        return {
            "interval": 5.0,
            "skip_modules": [
                "imagination",
                "counterfactual",
                "mental_simulator",
                "arch_searcher",
                "dream_consolidator",
            ],
        }
    if tier == 1:
        return {
            "interval": 2.0,
            "skip_modules": [
                "arch_searcher",
                "mental_simulator",
            ],
        }
    if tier >= 3:
        return {"interval": 1.0, "skip_modules": []}
    return {"interval": 2.0, "skip_modules": []}


def _sensor_config(tier: int) -> Dict[str, Any]:
    if tier == 0:
        return {
            "vision_model": None,
            "audio_model": None,
            "vision_interval": 2.0,
            "audio_interval": 2.0,
            "neural_perception": False,
        }
    if tier <= 2:
        return {
            "vision_model": "openai/clip-vit-base-patch32",
            "audio_model": "tiny",
            "vision_interval": 0.5,
            "audio_interval": 1.0,
            "neural_perception": True,
        }
    return {
        "vision_model": "openai/clip-vit-large-patch14",
        "audio_model": "base",
        "vision_interval": 0.25,
        "audio_interval": 0.5,
        "neural_perception": True,
    }


def _training_config(tier: int) -> Dict[str, Any]:
    if tier == 0:
        return {"enabled": False, "batch_size": 1, "lr_scale": 0.5}
    if tier == 1:
        return {"enabled": True, "batch_size": 4, "lr_scale": 1.0}
    if tier >= 3:
        return {"enabled": True, "batch_size": 32, "lr_scale": 1.0}
    return {"enabled": True, "batch_size": 16, "lr_scale": 1.0}


# ======================================================================
# Main config object
# ======================================================================


@dataclass
class ResourceConfig:
    tier: int = 1
    tier_name: str = "edge"

    hardware: Dict[str, Any] = field(default_factory=dict)
    encoder: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    cycle: Dict[str, Any] = field(default_factory=dict)
    sensor: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)

    boot_time: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_name": self.tier_name,
            "ram_gb": self.hardware.get("ram_gb"),
            "cpu_count": self.hardware.get("cpu_count"),
            "gpu": self.hardware.get("gpu", {}).get("name", "none"),
            "gpu_vram_gb": self.hardware.get("gpu", {}).get("vram_gb", 0),
            "api_backends": [k for k, v in self.hardware.get("api_keys", {}).items() if v],
            "gguf_models": len(self.hardware.get("gguf_models", [])),
            "encoder_mode": self.encoder.get("mode"),
            "llm_provider": self.llm.get("api_provider", "local"),
            "cycle_interval": self.cycle.get("interval"),
            "skip_modules": self.cycle.get("skip_modules", []),
            "boot_time": round(self.boot_time, 2),
        }

    def to_json(self) -> str:
        return json.dumps(self.summary(), indent=2, default=str)


_CONFIG: Optional[ResourceConfig] = None


def detect_resources(models_dir: Optional[str] = None) -> ResourceConfig:
    """Probe hardware and build the adaptive config. Called once at boot."""
    global _CONFIG
    t0 = time.time()

    if models_dir is None:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )

    hw: Dict[str, Any] = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "ram_gb": round(_get_ram_gb(), 1),
        "cpu_count": _get_cpu_count(),
        "gpu": _get_gpu_info(),
        "is_rpi": _is_raspberry_pi(),
        "api_keys": _detect_api_keys(),
        "gguf_models": _find_gguf_models(models_dir),
    }

    tier = _compute_tier(hw)

    cfg = ResourceConfig(
        tier=tier,
        tier_name=TIER_NAMES.get(tier, "unknown"),
        hardware=hw,
        encoder=_encoder_config(tier),
        llm=_llm_config(tier, hw),
        memory=_memory_config(tier),
        cycle=_cycle_config(tier),
        sensor=_sensor_config(tier),
        training=_training_config(tier),
        boot_time=round(time.time() - t0, 3),
    )

    _CONFIG = cfg
    return cfg


def get_resource_config() -> ResourceConfig:
    """Return the cached config, running detection if needed."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = detect_resources()
    return _CONFIG


def override_tier(tier: int):
    """Force a specific tier (for testing or manual override)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = detect_resources()

    _CONFIG.tier = tier
    _CONFIG.tier_name = TIER_NAMES.get(tier, "unknown")
    _CONFIG.encoder = _encoder_config(tier)
    _CONFIG.llm = _llm_config(tier, _CONFIG.hardware)
    _CONFIG.memory = _memory_config(tier)
    _CONFIG.cycle = _cycle_config(tier)
    _CONFIG.sensor = _sensor_config(tier)
    _CONFIG.training = _training_config(tier)
