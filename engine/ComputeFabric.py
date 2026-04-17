"""
ComputeFabric — Centralized device management for HaromaX6 (Phase 16).

Singleton that auto-detects GPU, manages device placement for all
neural modules, provides mixed-precision context managers, and offers
device-aware tensor factories.

When PyTorch is unavailable or the fabric is not initialized, all
operations fall back to CPU / no-op behavior transparently.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from contextlib import contextmanager

try:
    import torch
    import torch.nn as nn

    _TORCH = True
except (ImportError, OSError):
    _TORCH = False


class ComputeFabric:
    """Singleton managing device placement, mixed precision, and model registry."""

    _instance: Optional["ComputeFabric"] = None

    def __init__(self, force_cpu: bool = False):
        if not _TORCH:
            self._device = None
            self._use_gpu = False
            self._scaler = None
            self._registry: Dict[str, Any] = {}
            return

        if force_cpu or not torch.cuda.is_available():
            self._device = torch.device("cpu")
            self._use_gpu = False
        else:
            self._device = torch.device("cuda")
            self._use_gpu = True

        self._scaler = torch.cuda.amp.GradScaler() if self._use_gpu else None
        self._registry: Dict[str, nn.Module] = {}

    @classmethod
    def init(cls, force_cpu: bool = False) -> "ComputeFabric":
        cls._instance = cls(force_cpu=force_cpu)
        return cls._instance

    @classmethod
    def instance(cls) -> Optional["ComputeFabric"]:
        return cls._instance

    @property
    def device(self):
        if self._device is not None:
            return self._device
        if _TORCH:
            return torch.device("cpu")
        return None

    @property
    def use_gpu(self) -> bool:
        return self._use_gpu

    def register(self, name: str, module) -> Any:
        """Move module to device and track it."""
        if not _TORCH or self._device is None:
            return module
        module = module.to(self._device)
        self._registry[name] = module
        return module

    def tensor(self, data, dtype=None):
        if not _TORCH:
            return data
        dt = dtype if dtype is not None else torch.float32
        if isinstance(data, (list, tuple)) and len(data) > 0:
            el0 = data[0]
            if hasattr(el0, "__array__") and not isinstance(el0, (int, float, bool)):
                try:
                    import numpy as np

                    data = np.stack(data, axis=0)
                except (TypeError, ValueError):
                    pass
        t = torch.tensor(data, dtype=dt)
        if self._device is not None:
            t = t.to(self._device)
        return t

    def zeros(self, *shape, dtype=None):
        if not _TORCH:
            return None
        dt = dtype if dtype is not None else torch.float32
        t = torch.zeros(*shape, dtype=dt)
        if self._device is not None:
            t = t.to(self._device)
        return t

    def ones(self, *shape, dtype=None):
        if not _TORCH:
            return None
        dt = dtype if dtype is not None else torch.float32
        t = torch.ones(*shape, dtype=dt)
        if self._device is not None:
            t = t.to(self._device)
        return t

    @contextmanager
    def autocast(self):
        if self._use_gpu and _TORCH:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield

    def scale_loss(self, loss):
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def scaler_step(self, optimizer):
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def scaler_update(self):
        if self._scaler is not None:
            self._scaler.update()

    def stats(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "available": _TORCH,
            "device": str(self._device) if self._device else "none",
            "use_gpu": self._use_gpu,
            "registered_modules": len(self._registry),
        }
        if self._use_gpu and _TORCH:
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated(0) / 1024 / 1024, 1
                )
                info["gpu_memory_reserved_mb"] = round(
                    torch.cuda.memory_reserved(0) / 1024 / 1024, 1
                )
            except Exception as _e:
                print(f"[ComputeFabric] CUDA stats error: {_e}", flush=True)
        return info


def get_fabric() -> Optional[ComputeFabric]:
    """Return the singleton fabric, or None if not yet initialized."""
    return ComputeFabric.instance()
